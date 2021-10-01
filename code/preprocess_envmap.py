import os
import math

import numpy as np
import torch

import pyredner

from util.image_util import imwrite

ENVMAPS_ROOT = './datasets/envmaps/'
COEFFS_DIR = 'sh_coeffs/'
RASTERS_DIR = 'rasters/'

resolution = (256, 128)
device = torch.device('cuda:0')

def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    #deringed_coeffs[:, 9:9 + 7] += coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
    return deringed_coeffs

fn_list = os.listdir(os.path.join(ENVMAPS_ROOT, COEFFS_DIR))
for fn in fn_list:
    coeffs = torch.tensor(
            np.load(os.path.join(ENVMAPS_ROOT, COEFFS_DIR, fn)).transpose(),
            device=device)
    deringed_coeffs = deringing(coeffs, 6.0)
    envmap = pyredner.SH_reconstruct(deringed_coeffs, resolution)
    imwrite(envmap.cpu(),
            os.path.join(ENVMAPS_ROOT, RASTERS_DIR, '{}.exr'.format(fn.split('.')[0])))
