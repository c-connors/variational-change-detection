import theano
import numpy as np
from six.moves import cPickle


# Convenience class to hold image datasets.

class ImageDataset:

    def __init__(self, images, patch_shape, batch_size=None, shuffle=False):
        self.images, self.patch_shape, self.batch_size, self.shuffle = images, patch_shape, batch_size, shuffle
        self.patch_space_shape = ((images.shape[2] - patch_shape[0] + 1), (images.shape[3] - patch_shape[1] + 1))
        self.patch_count = np.prod(self.patch_space_shape)


# Convenience class to hold labeled datasets.

class LabeledDataset:

    def __init__(self, patches, labels, batch_size=None, shuffle=False, augment=False, aug_noise_std=0):
        self.patches, self.labels, self.batch_size, self.shuffle, self.augment, self.aug_noise_std = patches, labels, batch_size, shuffle, augment, aug_noise_std


# Sweeps a function over patches taken from a set of images. This is the main training and evaluation loop.

def image_iter(iter_fn, image_dataset,
        labeled_dataset=None,
        eval_fn=None,
        network_functions=None,
        valid_labeled_dataset=None,
        early_stop_rate=1,
        test_labeled_dataset=None,
        n_iter=None,
        give_results=False,
        history_rate=None,
        print_rate=None,
        metric_names=None):

    # Initialize and calculate default values
    if n_iter == None: n_iter = (image_dataset.patch_count - 1) // image_dataset.batch_size + 1
    combined_batch_size = image_dataset.batch_size + (0 if labeled_dataset == None else labeled_dataset.batch_size)
    batch = np.empty((combined_batch_size,) + image_dataset.images.shape[:2] + image_dataset.patch_shape, theano.config.floatX)
    if labeled_dataset != None:
        batch_labels = np.zeros((combined_batch_size,) + labeled_dataset.labels.shape[1:], theano.config.floatX)
        batch_label_present = (np.arange(combined_batch_size) >= image_dataset.batch_size).astype(theano.config.floatX)
    if history_rate != None: history = [None] * (1 + (n_iter - 1) // history_rate)
    results = None
    has_valid_or_test = valid_labeled_dataset != None or test_labeled_dataset != None
    best_valid_xent, test_at_best_valid_xent, best_network_functions_str = None, None, None

    # Run requested number of batches
    for i in range(n_iter + (1 if has_valid_or_test else 0)):

        # Evaluate on validation and test labeled sets if they were given
        eval_labeled_datasets = (valid_labeled_dataset, test_labeled_dataset)
        eval_xents = [None] * len(eval_labeled_datasets)
        for ii, eval_labeled_dataset in enumerate(eval_labeled_datasets):
            if eval_labeled_dataset != None:
                eval_xent = 0.
                for eval_batch_start in range(0, eval_labeled_dataset.patches.shape[0], eval_labeled_dataset.batch_size):
                    eval_batch_stop = min(eval_labeled_dataset.patches.shape[0], eval_batch_start + eval_labeled_dataset.batch_size)
                    eval_xent += (eval_batch_stop - eval_batch_start) * eval_fn(eval_labeled_dataset.patches[eval_batch_start:eval_batch_stop], eval_labeled_dataset.labels[eval_batch_start:eval_batch_stop])
                eval_xents[ii] = eval_xent / eval_labeled_dataset.patches.shape[0]
        valid_xent, test_xent = eval_xents

        # Early stopping with validation set
        if i % early_stop_rate == 0 and (best_valid_xent == None or valid_xent < best_valid_xent):
            best_network_functions_str = cPickle.dumps(network_functions)
            best_valid_xent = valid_xent
            test_at_best_valid_xent = test_xent

        if i == n_iter:
            if print_rate != None:
                # If training has ended, print the final validation and test evaluations
                if test_xent != None: print('End test result: %g' % test_xent)
                if best_valid_xent != None: print('Best validation result: %g' % best_valid_xent)
                if test_at_best_valid_xent != None: print('Test at best validation result: %g' % test_at_best_valid_xent)
        else:
            # Extract current batch from images
            batch_start = i * image_dataset.batch_size
            for ii in range(image_dataset.batch_size):
                idx = (batch_start + ii) % image_dataset.patch_count
                if image_dataset.shuffle:
                    if idx == 0: perm = np.random.permutation(image_dataset.patch_count)
                    idx = perm[idx]
                r_ind, c_ind = idx // image_dataset.patch_space_shape[1], idx % image_dataset.patch_space_shape[1]
                batch[ii] = image_dataset.images[:, :, r_ind:(r_ind + image_dataset.patch_shape[0]), c_ind:(c_ind + image_dataset.patch_shape[1])]
            batch_filled = min(image_dataset.patch_count, batch_start + image_dataset.batch_size) - batch_start

            # Extract current batch of labeled samples if labeled set was given
            if labeled_dataset == None: batch_args = (batch,)
            else:
                if labeled_dataset.augment: aug = np.random.randint(6, size=labeled_dataset.batch_size)
                for ii in range(labeled_dataset.batch_size):
                    idx = (batch_start + ii) % labeled_dataset.patches.shape[0]
                    if labeled_dataset.shuffle:
                        if idx == 0: labeled_perm = np.random.permutation(labeled_dataset.patches.shape[0])
                        idx = labeled_perm[idx]
                    patch = labeled_dataset.patches[idx]
                    batch_labels[image_dataset.batch_size + ii] = labeled_dataset.labels[idx]

                    # Apply data augmentation
                    if labeled_dataset.augment:
                        if aug[ii] == 0: patch = patch[:, :, ::-1] # Flip rows
                        elif aug[ii] == 1: patch = patch[:, :, :, ::-1] # Flip columns
                        elif aug[ii] == 2: patch = np.rot90(patch, axes=(2, 3)) # Rotate 90
                        elif aug[ii] == 3: patch = np.rot90(patch, k=2, axes=(2, 3)) # Rotate 180
                        elif aug[ii] == 4: patch = np.rot90(patch, k=3, axes=(2, 3)) # Rotate 270
                        patch = patch + np.random.normal(scale=labeled_dataset.aug_noise_std, size=patch.shape) # Add noise
                    batch[image_dataset.batch_size + ii] = patch
                batch_args = (batch, batch_labels, batch_label_present)

            # Apply function to current batch
            result = iter_fn(*batch_args)
            if give_results:
                if not isinstance(results, np.ndarray): results = np.empty(image_dataset.patch_space_shape + result.shape[1:], theano.config.floatX)
                results.reshape((results.shape[0] * results.shape[1],) + results.shape[2:])[batch_start:(batch_start + combined_batch_size)] = result[:batch_filled]

            # Optionally save and display results from current batch
            if history_rate != None and i % history_rate == 0: history[i // history_rate] = result
            if print_rate != None and i % print_rate == 0:
                print('Batch %d/%d' % (i + 1, n_iter))
                if metric_names != None:
                    for k, v in zip(metric_names, result): print('%s: %g' % (k, v.mean() if isinstance(v, np.ndarray) else v))
                    print('-' * 16)

    # Return a subsequence of [best_network_functions_str, results, history]
    ret_value = []
    if has_valid_or_test: ret_value.append(best_network_functions_str)
    if give_results: ret_value.append(results)
    if history_rate != None: ret_value.append(history)
    return ret_value[0] if len(ret_value) == 1 else ret_value
