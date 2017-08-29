#!/usr/bin/python3


'''
Change Detection by Auxiliary Variational Auto-Encoder.
'''


# Parse command line arguments
import os
import argparse
parser = argparse.ArgumentParser(description='Detect changes between two images.')
parser.add_argument('image_t0', help='path to the GeoTIFF image at t0')
parser.add_argument('image_t1', help='path to the GeoTIFF image at t1')
parser.add_argument('labels', help='path to the CSV labels')
default_save = 'last'
parser.add_argument('--save', '-s', help='name to save the trained model as', default=default_save)
parser.add_argument('--overwrite', '-o', action='store_true', help='allows overwriting of a previously saved model with the current one')
parser.add_argument('--load', '-l', help='name of a model to load and use instead of training')
parser.add_argument('--no-test', '-n', action='store_true', help='uses the entire label set for training instead of splitting out a test set')
parser.add_argument('--baseline', '-b', action='store_true', help='trains an SVM against raw RGB values for comparison')
args = parser.parse_args()
if args.save != default_save and not args.overwrite and os.path.isfile('interim/%s_model.pkl' % args.save): raise Exception('A model with this name was already saved. Provide the --overwrite flag (-o) when trying to --save over an existing model.')

import numpy as np
import scipy
import theano
from theano import tensor as T
from six.moves import cPickle
import sys
sys.setrecursionlimit(100000)
from theano.compile.nanguardmode import NanGuardMode
ngm = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)
from sklearn.svm import SVC

from vcd import image_iter, models, util


# Configuration
cfg = {

    # General
    'patch_shape': (33, 33),
    'aug_noise_std': 0.05,
    'train_test_split': 0.8, # Proportion of dataset to use for the train+validation set if the test set is enabled.
    'train_valid_split': 0.75, # Proportion of train+validation set to use for the training set.
    # Note that the final training set size is (label_count * train_test_split * train_valid_split).

    # M1 (unsupervised stage)
    'm1_n_iter': 50000,
    'm1_history_rate': 100,
    'm1_print_rate': 100,
    'm1_batch_size': 128,
    'm1_cnn_hidden_layers': 2,
    'm1_cnn_h_depth': 64,
    'm1_cnn_downscale': 2,
    'm1_cnn_res_depth': 5,
    'm1_mlp_hidden_layers': 2,
    'm1_mlp_n_h': 256,
    'm1_mlp_res_depth': 0,
    'm1_n_z': 16,
    'm1_q_z_ls_scale': 10000.,
    'm1_rec_ls_scale': 10000.,
    'm1_kld_cap': 0.25,
    'm1_kld_weight': 1.,
    'm1_rec_weight': 1.,
    'm1_learning_rate': 0.0001,

    # M2 (semi-supervised stage)
    'm2_n_iter': 30000,
    'm2_history_rate': 100,
    'm2_print_rate': 100,
    'm2_early_stop_rate': 100,
    'm2_eval_batch_size': 128,
    'm2_unlabeled_batch_size': 120,
    'm2_labeled_batch_size': 8,
    'm2_dropout': 0.,
    'm2_hidden_layers': 4,
    'm2_n_h': 128,
    'm2_n_z': 16,
    'm2_mlp_res_depth': 0,
    'm2_q_z_ls_scale': 10000.,
    'm2_rec_ls_scale': 10000.,
    'm2_p_k_scale': 1.,
    'm2_kld_cap': 0.25,
    'm2_kld_weight': 0.1,
    'm2_rec_weight': 0.1,
    'm2_cls_weight': 1.,
    'm2_learning_rate': 0.0001,
    'm2_save_gradients': False,

    # Classification (M1+M2)
    'cls_batch_size': 1024,
    'cls_print_rate': 1000,

    # Reconstruction generation
    'result_reconstruction_count': 256,
    'result_m1_save_latent_count': 10000,
    'result_m1_save_latent_batch_size': 256,

    # Postprocessing
    'closing_radius': 4,
    'opening_radius': 14
}
if not all([a % 2 == 1 for a in cfg['patch_shape']]): raise Exception('Only odd patch dimensions are supported')
parameterized_morph_filter = lambda raw_image: util.morph_filter(raw_image, cfg['closing_radius'], cfg['opening_radius'])


# Load data
m1_image_dataset, m2_image_dataset, cls_image_dataset, labeled_dataset, valid_labeled_dataset, test_labeled_dataset = util.load_data(args, cfg)
images = m1_image_dataset.images


# Either load the model,
if args.load != None:
    print('Loading model %s' % args.load)
    with open('interim/%s_model.pkl' % args.load, 'rb') as f: cur_model = cPickle.load(f)

# Or train it from scratch
else:
    # Create graph
    cur_model = {}
    metric_names, cur_model['network_functions'] = models.build_graph(images.shape[0], images.shape[1], labeled_dataset.labels.shape[2], cfg)

    # Train M1 (unsupervised stage)
    print('Training M1')

    history = image_iter.image_iter(cur_model['network_functions']['m1_train'],
            m1_image_dataset,
            n_iter=cfg['m1_n_iter'],
            history_rate=cfg['m1_history_rate'],
            print_rate=cfg['m1_print_rate'],
            metric_names=metric_names['m1_train'])

    with open('interim/%s_m1_metrics.pkl' % args.save, 'wb') as f: cPickle.dump({'history': history, 'names': metric_names['m1_train']}, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # Train M2 (semi-supervised stage)
    print('Training M2')
    m2_full_train_fn = lambda batch, batch_labels, batch_label_present: cur_model['network_functions']['m2_train'](cur_model['network_functions']['m1_latent'](batch), batch_labels, batch_label_present)
    m2_full_eval_fn = lambda batch, batch_labels: cur_model['network_functions']['m2_eval'](cur_model['network_functions']['m1_latent'](batch), batch_labels)

    best_network_functions_str, history = image_iter.image_iter(m2_full_train_fn,
            m2_image_dataset,
            labeled_dataset=labeled_dataset,
            eval_fn=m2_full_eval_fn,
            network_functions=cur_model['network_functions'],
            valid_labeled_dataset=valid_labeled_dataset,
            early_stop_rate=cfg['m2_early_stop_rate'],
            test_labeled_dataset=test_labeled_dataset,
            n_iter=cfg['m2_n_iter'],
            history_rate=cfg['m2_history_rate'],
            print_rate=cfg['m2_print_rate'],
            metric_names=metric_names['m2_train'])

    best_network_functions = cPickle.loads(best_network_functions_str)
    cur_model['network_functions'].update(best_network_functions)
    with open('interim/%s_m2_metrics.pkl' % args.save, 'wb') as f: cPickle.dump({'history': history, 'names': metric_names['m1_train']}, f, protocol=cPickle.HIGHEST_PROTOCOL)


# Generate change probabilities of train and validation sets
print('Generating change probabilities of train and validation sets')
r_rad, c_rad = cfg['patch_shape'][0] // 2, cfg['patch_shape'][1] // 2
train_valid_patches = np.concatenate((labeled_dataset.patches, valid_labeled_dataset.patches))
train_valid_rgb_patches = train_valid_patches[:, :, :, r_rad, c_rad].reshape(train_valid_patches.shape[0], -1)
train_valid_t_rgb_patches = train_valid_patches[:, :, :, r_rad, c_rad].reshape(train_valid_patches.shape[0], train_valid_patches.shape[1], -1)
train_valid_labels = np.concatenate((labeled_dataset.labels, valid_labeled_dataset.labels))
train_valid_labels_numeric = np.argmax(train_valid_labels, axis=2)
train_valid_change_labels = np.stack([train_valid_labels_numeric[:, i] != train_valid_labels_numeric[:, i + 1] for i in range(train_valid_labels_numeric.shape[1] - 1)], 1)
train_valid_label_probabilities = np.empty(train_valid_labels.shape)
for batch_start in range(0, train_valid_labels.shape[0], cfg['cls_batch_size']):
    batch_stop = batch_start + cfg['cls_batch_size']
    train_valid_label_probabilities[batch_start:batch_stop] = cur_model['network_functions']['m2_cls'](cur_model['network_functions']['m1_latent'](train_valid_patches[batch_start:batch_stop]))
train_valid_change_probabilities = np.stack([np.sum([train_valid_label_probabilities[:, i, ii] * (1 - train_valid_label_probabilities[:, i + 1, ii]) for ii in range(train_valid_labels.shape[2])], 0) for i in range(train_valid_label_probabilities.shape[1] - 1)], 1)


# Generate change probabilities of test set if there is one
if not args.no_test:
    # Get VAE change probabilities of test set
    print('Generating change probabilities of test set')
    test_patches, test_labels = test_labeled_dataset.patches, test_labeled_dataset.labels
    test_rgb_patches = test_patches[:, :, :, r_rad, c_rad].reshape(test_patches.shape[0], -1)
    test_t_rgb_patches = test_patches[:, :, :, r_rad, c_rad].reshape(test_patches.shape[0], test_patches.shape[1], -1)
    test_labels_numeric = np.argmax(test_labels, axis=2)
    test_change_labels = np.stack([test_labels_numeric[:, i] != test_labels_numeric[:, i + 1] for i in range(test_labels_numeric.shape[1] - 1)], 1)
    test_label_probabilities = np.empty(test_labels.shape)
    for batch_start in range(0, test_labels.shape[0], cfg['cls_batch_size']):
        batch_stop = batch_start + cfg['cls_batch_size']
        test_label_probabilities[batch_start:batch_stop] = cur_model['network_functions']['m2_cls'](cur_model['network_functions']['m1_latent'](test_labeled_dataset.patches[batch_start:batch_stop]))
    test_change_probabilities = np.stack([np.sum([test_label_probabilities[:, i, ii] * (1 - test_label_probabilities[:, i + 1, ii]) for ii in range(test_labels.shape[2])], 0) for i in range(test_label_probabilities.shape[1] - 1)], 1)

    # Generate CSV of predictions
    predictions = np.concatenate((test_change_labels, test_change_probabilities), 1)
    predictions_header = ','.join(['actual_%d' % i for i in range(test_change_labels.shape[1])] + ['prediction_%d' % i for i in range(test_change_probabilities.shape[1])])
    predictions_fmt = ['%d'] * test_change_labels.shape[1] + ['%g'] * test_change_probabilities.shape[1]
    with open('results/predictions.csv', 'wb') as f: np.savetxt(f, predictions, fmt=predictions_fmt, delimiter=',', header=predictions_header, comments='')


# Train VAE SVMs if a previous model was not loaded
if args.load == None:
    cur_model['vae_svm'], cur_model['vae_rgb_svm'] = SVC(), SVC()
    cur_model['vae_svm'].fit(train_valid_change_probabilities, np.any(train_valid_change_labels, 1))
    cur_model['vae_rgb_svm'].fit(np.concatenate((train_valid_change_probabilities, train_valid_rgb_patches), 1), np.any(train_valid_change_labels, 1))

    # Score VAE SVMs if there is a test set
    if not args.no_test:
        print('vae_svm: %g' % cur_model['vae_svm'].score(test_change_probabilities, np.any(test_change_labels, 1)))
        print('vae_rgb_svm: %g' % cur_model['vae_rgb_svm'].score(np.concatenate((test_change_probabilities, test_rgb_patches), 1), np.any(test_change_labels, 1)))


# Classify entire image using VAE and VAE+RGB SVMs
print('Detecting changes across entire image')
full_cls_fn = lambda batch: cur_model['network_functions']['m2_cls'](cur_model['network_functions']['m1_latent'](batch))
image_label_probabilities = image_iter.image_iter(full_cls_fn, cls_image_dataset, give_results=True, print_rate=cfg['cls_print_rate'])
change_probabilities = np.stack([np.sum([image_label_probabilities[:, :, i, ii] * (1 - image_label_probabilities[:, :, i + 1, ii]) for ii in range(train_valid_labels.shape[2])], 0) for i in range(image_label_probabilities.shape[2] - 1)], 2)
change_probabilities_padded_flat = np.pad(change_probabilities, ((cfg['patch_shape'][0] // 2, cfg['patch_shape'][0] // 2), (cfg['patch_shape'][1] // 2, cfg['patch_shape'][1] // 2), (0, 0)), 'constant').reshape((-1, change_probabilities.shape[2]))
vae_svm_pred = cur_model['vae_svm'].predict(change_probabilities_padded_flat).reshape((images.shape[2], images.shape[3]))
images_rgb_flat = images.transpose(2, 3, 0, 1).reshape((images.shape[2] * images.shape[3], -1))
vae_rgb_svm_pred = cur_model['vae_rgb_svm'].predict(np.concatenate((change_probabilities_padded_flat, images_rgb_flat), 1)).reshape((images.shape[2], images.shape[3]))
util.save_result_images('change_vae_svm', vae_svm_pred, images, filter_fn=parameterized_morph_filter)


# Compare with baseline if requested
if args.baseline:

    # Train baseline SVMs
    rgb_change_svm, rgb_class_svms = SVC(), [SVC() for i in range(images.shape[0])]
    rgb_change_svm.fit(train_valid_rgb_patches, np.any(train_valid_change_labels, 1))
    for i, svm in enumerate(rgb_class_svms): svm.fit(train_valid_t_rgb_patches[:, i], train_valid_labels_numeric[:, i])

    # Score baseline SVMs if there is a test set
    if not args.no_test:
        rgb_class_svm_pred = np.stack([svm.predict(test_t_rgb_patches[:, i]) for i, svm in enumerate(rgb_class_svms)], 1)
        rgb_class_svm_change_pred = np.stack([rgb_class_svm_pred[:, i] != rgb_class_svm_pred[:, i + 1] for i in range(rgb_class_svm_pred.shape[1] - 1)], 1)
        print('rgb_change_svm: %g' % rgb_change_svm.score(test_rgb_patches, np.any(test_change_labels, 1)))
        print('rgb_class_svms: %g' % (rgb_class_svm_change_pred != test_change_labels).mean())

    # Classify entire image using baseline SVMs
    rgb_change_svm_pred = rgb_change_svm.predict(images_rgb_flat).reshape((images.shape[2], images.shape[3]))
    util.save_result_images('change_rgb_svm', rgb_change_svm_pred, images, filter_fn=parameterized_morph_filter)


# Save the model if a new name was given
if args.save != args.load:
    with open('interim/%s_model.pkl' % args.save, 'wb') as f: cPickle.dump(cur_model, f, protocol=cPickle.HIGHEST_PROTOCOL)


print('Done')
