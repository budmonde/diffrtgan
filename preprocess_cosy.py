import os
import math
import numpy as np
import skimage
import skimage.morphology
from PIL import Image

from util.image_util import imread, imwrite

ROOT_DIR = './datasets/cosy/'
RAW_IMG_DIR = os.path.join(ROOT_DIR, 'raw/img')
LABEL_DIR = os.path.join(ROOT_DIR, 'raw/label')
MASK_DIR = os.path.join(ROOT_DIR, 'mask')
IMG_DIR = os.path.join(ROOT_DIR, 'img')


if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

def center_and_scale(img, mask):
    # Scale the Image
    props = skimage.measure.regionprops(mask)

    # Reject unless there's only one blob
    if (len(props) !=1):
        print('props {}'.format(len(props)), end=' ')
        return None

    # Reject if bbox ratio is off
    bbox = props[0].bbox
    ratio = (bbox[4] - bbox[1]) / (bbox[3] - bbox[0])
    # Tuned magic numbers
    RATIO_MEAN = 2.83
    RATIO_STD = 0.05
    if abs(ratio - RATIO_MEAN) > RATIO_STD:
        print('ratio {:4f}'.format(ratio), end=' ')
        return None

    # Normalize car scale
    image_size = mask.shape[0] * mask.shape[1]
    BASELINE_PERCENTAGE = 0.4
    target_area = image_size * BASELINE_PERCENTAGE
    mask_area = props[0].filled_area

    factor = math.sqrt(target_area / mask_area)
    scale = skimage.transform.AffineTransform(scale=(1/factor, 1/factor))
    mask = skimage.transform.warp(mask, scale, order=0, preserve_range=True)
    img = skimage.transform.warp(img, scale, order=1, preserve_range=True)

    dx = (factor - 1) * mask.shape[0] / 2
    dy = (factor - 1) * mask.shape[1] / 2

    trans = skimage.transform.AffineTransform(translation=(dx, dy))
    mask = skimage.transform.warp(mask, trans, order=0, preserve_range=True)
    img = skimage.transform.warp(img, trans, order=1, preserve_range=True)

    # Translate the Image
    props = skimage.measure.regionprops(mask.astype('uint8'))

    bbox = props[0].bbox
    cy = (bbox[3] + bbox[0]) / 2
    dy = int(round((cy - mask.shape[0]/2)))

    tfm = skimage.transform.AffineTransform(translation=(0, dy))
    mask = skimage.transform.warp(mask, tfm, order=0, preserve_range=True)
    img = skimage.transform.warp(img, tfm, order=1, preserve_range=True)

    return (img, mask)

fn_list = os.listdir(RAW_IMG_DIR)
for fn in fn_list:
    # Fetch file metadata
    fn = fn.split('.')[0]
    print('Processing: {}'.format(fn), end='\t')
    img_fpath = os.path.join(RAW_IMG_DIR, '{}.exr'.format(fn))
    label_fpath = os.path.join(LABEL_DIR, 'label_{}.png'.format(fn))

    # Open images. Open label using PIL because image is encoded in uint8
    img = imread(img_fpath)
    label = np.array(Image.open(label_fpath))

    # Isolate masks
    label[label != 250] = 0.
    label[label == 250] = 1.

    # Clean up the mask
    label = label.astype(bool)
    skimage.morphology.remove_small_objects(label, min_size = 1000, in_place=True)
    skimage.morphology.remove_small_holes(label, area_threshold = 1000, in_place=True)
    label = label.astype('uint8')

    # Normalize the mask
    out = center_and_scale(img, label)

    # Reject if there was an issue with the image
    if out == None:
        print('Rejecting: {}'.format(fn))
        continue

    # Write to disk
    print('Writing: {}'.format(fn))
    imwrite(out[0], os.path.join(IMG_DIR, '{}.png'.format(fn)))
    imwrite(out[1], os.path.join(MASK_DIR, 'label_{}.png'.format(fn)))
