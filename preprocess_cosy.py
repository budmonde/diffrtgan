import os
import math
import numpy as np
import skimage
import skimage.morphology
from PIL import Image

from image_util import imread, imwrite

ROOT_DIR = './datasets/cosy/'
IMG_DIR = '{}img/'.format(ROOT_DIR)
LABEL_DIR = '{}label/'.format(ROOT_DIR)
MASK_DIR = '{}mask/'.format(ROOT_DIR)
IMG_OUT_DIR = '{}img_out/'.format(ROOT_DIR)

if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)
if not os.path.exists(IMG_OUT_DIR):
    os.makedirs(IMG_OUT_DIR)

def center_and_scale(img, mask):
    # Scale the Image
    props = skimage.measure.regionprops(mask)

    if (len(props) < 1):
        return None

    area = props[0].filled_area
    baseline_area = 20000
    s = math.sqrt(area / baseline_area)
    tfm = skimage.transform.AffineTransform(scale=(s, s))
    mask = skimage.transform.warp(mask, tfm, order=0, preserve_range=True)
    img = skimage.transform.warp(img, tfm, order=1, preserve_range=True)

    # Translate the Image
    props = skimage.measure.regionprops(mask.astype('uint8'))

    if (len(props) < 1):
        return None

    bbox = props[0].bbox
    cx, cy = (bbox[4] + bbox[1]) / 2,\
             (bbox[3] + bbox[0]) / 2
    dx, dy = int(round((cx - mask.shape[0]/2))),\
             int(round((cy - mask.shape[1]/2)))

    tfm = skimage.transform.AffineTransform(translation=(dx, dy))
    mask = skimage.transform.warp(mask, tfm, order=0, preserve_range=True)
    img = skimage.transform.warp(img, tfm, order=1, preserve_range=True)

    return (img, mask)

dist_arr = []

fn_list = os.listdir(IMG_DIR)
for fn in fn_list:
    fn = fn.split('.')[0]
    #if (fn != '0_5_frame_8373293'):
    #    continue
    print('Processing: {}'.format(fn), end=' ')
    img_fpath = os.path.join(IMG_DIR, '{}.exr'.format(fn))
    img = imread(img_fpath)
    label_fpath = os.path.join(LABEL_DIR, 'label_{}.png'.format(fn))
    label = np.array(Image.open(label_fpath))
    label[label != 250] = 0.
    label[label == 250] = 1.

    label = label.astype(bool)
    skimage.morphology.remove_small_objects(label, min_size = 200, in_place=True)
    skimage.morphology.remove_small_holes(label, in_place=True)
    label = label.astype('uint8')

    mean = np.mean(label)
    if mean <= 0.1:
        print('\tRejecting: {}'.format(fn))
        continue

    out = center_and_scale(img, label)

    if out == None:
        print('\tRejecting: {}'.format(fn))
        continue

    print('\tWriting: {}'.format(fn))
    dist_arr.append(mean)
    imwrite(out[0], os.path.join(IMG_OUT_DIR, '{}.png'.format(fn)))
    imwrite(out[1], os.path.join(MASK_DIR, 'label_{}.png'.format(fn)))

print(np.histogram(np.array(dist_arr)))
