import os

import torch

import pyredner

from util.render_util import LearnMesh
from util.misc_util import *

pyredner.set_use_gpu(False)
pyredner.set_device(torch.device('cpu'))

MESHES_PATH = './datasets/meshes/full'
OUTPUT_PATH = './datasets/meshes/serialized'

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    mesh_path_list = get_child_paths(MESHES_PATH, ext='obj')
    for path in mesh_path_list:
        print(f'Serializing mesh {path}')
        mesh = LearnMesh.load_obj(path)
        torch.save(mesh.state_dict(),
                os.path.join(OUTPUT_PATH, f'{get_fn(path)}.pth'))

if __name__ == '__main__':
    main()
