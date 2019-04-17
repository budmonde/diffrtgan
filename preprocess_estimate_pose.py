import argparse
import json
import math
import os

import numpy as np
import skimage
import skimage.morphology

from util.image_util import imread
from util.misc_util import *


# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets/cosy/', help='Path to dataset for pose estimation')
parser.add_argument('--resolution', type=int, default=256, help='resolution')
parser.add_argument('--pose_bank_path', type=str, default='./datasets/poses/octavia_clean', help='Path for pose bank to use')
opt = parser.parse_args()

def crop_and_rescale(img):
    img = img.astype('uint8')
    props = skimage.measure.regionprops(img)
    bbox = props[0].bbox
    img = img[bbox[0]:bbox[3],bbox[1]:bbox[4],:]
    img = skimage.transform.resize(img,
        (opt.resolution, opt.resolution), order = 0)
    return img

# Load pose bank config
with open(os.path.join(opt.pose_bank_path, 'data.json'), 'r') as f:
    data = json.load(f)

# Load images to estimate poses for
fpath_list = get_child_paths(os.path.join(opt.dataroot, 'mask'))

# Initialize objects before iteration
best_pose_dict = dict()

for fpath in fpath_list:
    image = crop_and_rescale(imread(fpath))

    best_pose = dict()
    best_loss = float("inf")

    key = get_fn(fpath)

    print("Processing {}".format(key), end='\t')
    for k, v in data.items():
        pose_path = os.path.join(opt.pose_bank_path, '{}.png'.format(k))
        pose_img = crop_and_rescale(imread(pose_path)[:,:,0:1])

        l2 = np.sum((image - pose_img)**2)
        if l2 < best_loss:
            best_pose['geo_rotation'] = [v['azimuth'], v['elevation'], 0.0]
            best_loss = l2

    print("Best pose: {}".format(best_pose))
    best_pose_dict[key] = best_pose

with open(os.path.join(opt.dataroot, 'poses.json'), 'w') as metafile:
    json.dump(best_pose_dict, metafile)
