import argparse
import json
import math
import os

import numpy as np

from util.image_util import imread
from util.transform_util import Resize


# Argparser
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets/cosy/', help='Path to dataset for pose estimation')
parser.add_argument('--data_resolution', type=int, default=256, help='Resolution')
parser.add_argument('--pose_bank_path', type=str, default='./datasets/poses/octavia_clean/100_10', help='Path for pose bank to use')
opt = parser.parse_args()

# Get granularity of pose bank
poses_dirname = opt.pose_bank_path.split('/')[-1]
num_azimuth, num_distance = (int(i) for i in poses_dirname.split('_'))

# Load images to estimate poses for
masks_path = os.path.join(opt.dataroot, 'mask')
fn_list = os.listdir(masks_path)

# Initialize objects before iteration
resizer = Resize(opt.data_resolution, order = 0)
best_pose_dict = dict()

for fn in fn_list:
    fpath = os.path.join(masks_path, fn)

    image = resizer(imread(fpath)[:,:,0:1])
    best_pose = (0, 0)
    best_loss = float("inf")

    key = fn.split('.')[0]
    print("Processing {}".format(key), end='\t')
    for i in range(num_azimuth):
        for j in range(num_distance):
            pose_path = os.path.join(opt.pose_bank_path, '{}_{}.png'.format(i, j))
            pose_img = imread(pose_path)[:,:,0:1]

            l2 = np.sum((image - pose_img)**2)
            if l2 < best_loss:
                azimuth = 2 * math.pi * i / num_azimuth
                distance = 6 + 4 * j / num_distance
                best_pose = (azimuth, distance)
                best_loss = l2

    print("Best pose: {}".format(best_pose))
    best_pose_dict[key] = best_pose

with open(os.path.join(opt.dataroot, 'poses.json'), 'w') as metafile:
    json.dump(best_pose_dict, metafile)
