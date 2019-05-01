import argparse
import json
import os
import subprocess

from skimage.transform import rotate

from util.image_util import imread, imwrite
from util.misc_util import *
from util.sample_util import *


BLENDER = '/home/budmonde/opt/blender/blender'
OBJ_DIR = './datasets/meshes/learn'
OUT_PATH = './datasets/textures/bake'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

sampler = RGBFileSamplerFactory('./datasets/distributions/diffuse.txt')

data = dict()
for in_path in get_child_paths(OBJ_DIR, ext='obj'):
    mesh_name = get_fn(path)
    out_path = os.path.join(OUT_PATH, f'{mesh_name}.png')
    data[mesh_name] = sampler()
    subprocess.run([
        BLENDER,
            '--background',
            '--python', 'blender/add_dirt_blender.py',
            '--',
                '--input_path', in_path,
                '--output_path', out_path,
                '--albedo_r', str(data[path][0]),
                '--albedo_g', str(data[path][1]),
                '--albedo_b', str(data[path][2])],
        check=True)

    # Post process output
    image = imread(out_path)
    image = rotate(image, 180)
    imwrite(image, out_path)

    with open(os.path.join(OUT_PATH, 'data.json'), 'w') as f:
        json.dump(data, f)
