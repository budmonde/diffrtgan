import argparse
import json
import os
import subprocess

from util.misc_util import *
from util.sample_util import *


BLENDER = '/home/budmonde/opt/blender/blender'
OBJ_DIR = './datasets/meshes/learn'
OUT_PATH = './datasets/textures/bake'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

sampler = RGBFileSamplerFactory('./datasets/distributions/diffuse.txt')

data = dict()
for path in get_child_paths(OBJ_DIR, ext='obj'):
    data[path] = sampler()
    subprocess.run([
        BLENDER,
            '--background',
            '--python', 'blender/add_dirt_blender.py',
            '--',
                '--input_path', path,
                '--output_path', OUT_PATH,
                '--albedo_r', str(data[path][0]),
                '--albedo_g', str(data[path][1]),
                '--albedo_b', str(data[path][2])],
        check=True)

    with open(os.path.join(OUT_PATH, 'data.json'), 'w') as f:
        json.dump(data, f)
