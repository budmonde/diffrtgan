import argparse
from datetime import datetime
import json
import os
import random
import time

import torch

from util.image_util import imread, imwrite
from util.render_util import RenderLogger, Render


# Load arguments
parser = argparse.ArgumentParser()
# Scene args
parser.add_argument('--meshes_path', type=str, default='./datasets/meshes/one')
parser.add_argument('--envmaps_path', type=str, default='./datasets/envmaps/rasters')
parser.add_argument('--texture_path', type=str, default='./datasets/textures/decal.png')
# Render config args
parser.add_argument('--fineSize', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=200)
parser.add_argument('--max_bounces', type=int, default=1)
parser.add_argument('--label', type=str, default='debug')
# Output args
parser.add_argument('--num_imgs', type=int, default=100)
parser.add_argument('--root_path', type=str, default='./datasets/renders/')
# Misc
parser.add_argument('--gpu_id', type=int, default=0, help='CUDA GPU id used for rendering. -1 for CPU')
opt = parser.parse_args()


# Pre-process arguments
channels = [
    'radiance',
    'alpha',
    'position',
    'shading_normal',
    'diffuse_reflectance'
]
now = datetime.now()
now_string = '{}-{}_{}-{}'.format(now.month, now.day, now.hour, now.minute)
subdir_name = 'res{}mc{}_{}_{}'.format(opt.fineSize, opt.num_samples, opt.label, now_string)
out_path = os.path.join(opt.root_path, subdir_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)
device = torch.device('cuda:{}'.format(opt.gpu_id) if opt.gpu_id != -1 else 'cpu')


# Render setup
texture = torch.tensor(imread(opt.texture_path), dtype=torch.float32, device=device)
render_logger = RenderLogger()
renderer = Render(
    opt.meshes_path,
    opt.envmaps_path,
    opt.fineSize,
    opt.num_samples,
    opt.max_bounces,
    device,
    channels = channels,
    logger = render_logger)

with open(os.path.join(out_path, 'data.bak'), 'a+') as backupfile:
    for i in range(opt.num_imgs):
        iter_start_time = time.time()
        out = renderer(texture)
        key = render_logger.get_active_id()
        print('Generated Image: #\t{} -- {} in {}'.format(i, key, time.time() - iter_start_time))
        imwrite(out[:, :,   : 3], os.path.join(out_path, 'img',      '{}.png'.format(key)))
        imwrite(out[:, :,  3: 4], os.path.join(out_path, 'mask',     '{}.png'.format(key)))
        imwrite(out[:, :,  4: 7], os.path.join(out_path, 'position', '{}.png'.format(key)))
        imwrite(out[:, :,  7:10], os.path.join(out_path, 'normal',   '{}.png'.format(key)))
        imwrite(out[:, :, 10:13], os.path.join(out_path, 'albedo',   '{}.png'.format(key)))
        json.dump({key: render_logger[key]}, backupfile)

with open(os.path.join(out_path, 'data.json'), 'w') as metafile:
    json.dump(render_logger.get_all(), metafile)
