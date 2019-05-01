import argparse
import os
import subprocess

BLENDER = '/home/budmonde/opt/blender/blender'
OBJ_DIR = '/home/budmonde/dev/rendergan/datasets/meshes/learn'
OUT_PATH = './bake'

def get_child_paths(path, ext=None):
    paths = [os.path.join(path, fn) for fn in os.listdir(path)]
    paths = list(filter(lambda fn: fn.split('.')[-1] == ext, paths))
    return paths

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

for path in get_child_paths(OBJ_DIR, 'obj'):
    subprocess.run([
        BLENDER,
            '--background',
            '--python', 'add_dirt_blender.py',
            '--',
                '--input_path', path,
                '--output_path', OUT_PATH],
        check=True)
