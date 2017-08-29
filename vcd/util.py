'''
Utility functions.
'''

import numpy as np
import gdal
import scipy
import scipy.misc
from scipy.ndimage.morphology import grey_opening, grey_closing
import theano
import sklearn.model_selection


from . import image_iter


'''
Load images from GeoTIFF.
'''
def load_images(fnames):
    images, geotransforms = [], []
    for fname in fnames:
        dataset = gdal.Open(fname)
        images.append(dataset.ReadAsArray().astype(theano.config.floatX))
        geotransforms.append(dataset.GetGeoTransform())
    if not all([a.shape[0] == images[0].shape[0] for a in images[1:]]): raise Exception('Images have different band counts')
    min_r_image, min_c_image = [np.argmin([a.shape[i] for a in images]) for i in (1, 2)]
    if not min_r_image == min_c_image: raise Exception('Image with smallest width must be image with smallest height')
    gt = geotransforms[min_r_image] # Use geotransform of smallest image
    images = np.stack([np.stack([scipy.misc.imresize(a[i], (images[0].shape[1], images[0].shape[2]), mode='F') for i in range(a.shape[0])]) for a in images]) # Rescale to smallest image
    images = (images - images.mean((2, 3), keepdims=True)) / images.std((2, 3), keepdims=True) # Normalize
    return images, gt


'''
Load labels.
'''
def load_labels(fname):
    return np.loadtxt(fname, delimiter=',', skiprows=1, dtype=theano.config.floatX)


'''
Load data and construct datasets.
'''
def load_data(args, cfg):
    images, gt = load_images((args.image_t0, args.image_t1))
    print('Loaded %d images with size (%d, %d) and %d channels' % (images.shape[0], images.shape[2], images.shape[3], images.shape[1]))
    coordinates_and_labels = load_labels(args.labels)
    patches = np.stack([get_geo_slices(coordinates_and_labels[i, 0], coordinates_and_labels[i, 1], cfg['patch_shape'][0] // 2, cfg['patch_shape'][1] // 2, gt, images) for i in range(coordinates_and_labels.shape[0])])
    labels_numeric = coordinates_and_labels[:, 2:] # Each sample and timestep labeled with its class index as an integer
    change_labels = np.stack([labels_numeric[:, i] != labels_numeric[:, i + 1] for i in range(labels_numeric.shape[1] - 1)], 1)
    n_class = int(labels_numeric.max() + 1)
    labels = np.arange(n_class)[np.newaxis, np.newaxis] == labels_numeric[:, :, np.newaxis] # Each sample and timestep given an indicator label that is zero everywhere except at the index of its class
    if args.no_test: train_valid_patches, train_valid_labels, train_valid_change_labels, train_valid_labels_numeric = patches, labels, change_labels, labels_numeric
    else: train_valid_patches, test_patches, train_valid_labels, test_labels, train_valid_change_labels, test_change_labels, train_valid_labels_numeric, test_labels_numeric = sklearn.model_selection.train_test_split(patches, labels, change_labels, labels_numeric, train_size=cfg['train_test_split'], stratify=change_labels.any(1))
    train_patches, valid_patches, train_labels, valid_labels, train_change_labels, valid_change_labels = sklearn.model_selection.train_test_split(train_valid_patches, train_valid_labels, train_valid_change_labels, train_size=cfg['train_valid_split'], stratify=train_valid_change_labels.any(1))
    print('Loaded %d labels from %d classes across %d timesteps:' % (labels.shape[0], labels.shape[2], labels.shape[1]))
    print('  Training set: %d labels' % train_labels.shape[0])
    print('  Validation set: %d labels' % valid_labels.shape[0])
    if args.no_test: print('  Test set disabled')
    else: print('  Test set: %d labels' % test_labels.shape[0])

    m1_image_dataset = image_iter.ImageDataset(images, cfg['patch_shape'], batch_size=cfg['m1_batch_size'], shuffle=True)
    m2_image_dataset = image_iter.ImageDataset(images, cfg['patch_shape'], batch_size=cfg['m2_unlabeled_batch_size'], shuffle=True)
    cls_image_dataset = image_iter.ImageDataset(images, cfg['patch_shape'], batch_size=cfg['cls_batch_size'])
    labeled_dataset = image_iter.LabeledDataset(train_patches, train_labels, batch_size=cfg['m2_labeled_batch_size'], shuffle=True, augment=True, aug_noise_std=cfg['aug_noise_std'])
    valid_labeled_dataset = image_iter.LabeledDataset(valid_patches, valid_labels, batch_size=cfg['m2_eval_batch_size'])
    test_labeled_dataset = None if args.no_test else image_iter.LabeledDataset(test_patches, test_labels, batch_size=cfg['m2_eval_batch_size'])
    return m1_image_dataset, m2_image_dataset, cls_image_dataset, labeled_dataset, valid_labeled_dataset, test_labeled_dataset


'''
Geo coordinates to pixel coordinates.
'''
def geo_to_pixel(x, y, gt):
    if gt[2] != 0 or gt[4] != 0: raise Exception('Projections with rotation/skew are not supported')
    return (x - gt[0]) / gt[1], (y - gt[3]) / gt[5]


'''
Get slices of a set of images from geo coordinates.
'''
def get_geo_slices(x, y, r_rad, c_rad, gt, images):
    c, r = [np.int64(a.round()) for a in geo_to_pixel(x, y, gt)]
    data = [a[:, max(0, r - r_rad):(r + r_rad + 1), max(0, c - c_rad):(c + c_rad + 1)] for a in images]
    return np.stack([np.pad(a, ((0, 0), (max(0, r_rad - r), max(0, r + r_rad + 1 - images[0].shape[1])), (max(0, c_rad - c), max(0, c + c_rad + 1 - images[0].shape[2]))), 'linear_ramp') for a in data])


'''
Morphological closing followed by opening to fill small negative regions and
remove small positive regions, respectively. This is intended to remove thin
positive regions along physical class boundaries (edges of buildings, etc.) as
we know we are only looking for changes which are larger than a certain area.
'''
def morph_filter(raw_image, closing_radius, opening_radius):

    # Create circular masks
    close_footprint, open_footprint = [[[0 for ii in range(r * 2 + 1)] for i in range(r * 2 + 1)] for r in [closing_radius, opening_radius]]
    for fp in [close_footprint, open_footprint]:
	    r = (len(fp) - 1) / 2
	    for i in range(len(fp)):
		    for ii in range(len(fp)):
			    if (i - r) ** 2 + (ii - r) ** 2 <= r ** 2: fp[i][ii] = 1

    # Perform filtering
    filtered_image = raw_image
    filtered_image = grey_closing(filtered_image, footprint=close_footprint)
    filtered_image = grey_opening(filtered_image, footprint=open_footprint)
    return filtered_image


'''
Save result images with optional filtering.
'''
def save_result_images(name, result, images, filter_fn=None):

    # Save result image by itself and as an overlay on the original images.
    scipy.misc.imsave('results/%s.png' % name, result)
    for i, image in enumerate(images):
        scipy.misc.imsave('results/%s_overlay_t%d.png' % (name, i), (np.clip(image, -2., 2.) + 3 * np.array([1, 0, 0])[:, np.newaxis, np.newaxis] * result[np.newaxis]).transpose(1, 2, 0))

    # Save filtered versions if a filter function is provided.
    if filter_fn != None:
        filtered_name = '%s_filtered' % name
        filtered_result = filter_fn(result)
        save_result_images(filtered_name, filtered_result, images, filter_fn=None)
