import argparse
from datetime import datetime
import hashlib
import os
import random
import json

import torch

from util.image_util import imread, imwrite
from util.render_util import RenderLogger, Render


# Load arguments
parser = argparse.ArgumentParser()
# Scene args
parser.add_argument('--meshes_path', type=str, default="./datasets/meshes/one_mtl")
parser.add_argument('--envmaps_path', type=str, default="./datasets/envmaps/rasters")
parser.add_argument('--texture_path', type=str, default="./datasets/textures/checkerboard/train/checkerboard.png")
# Render config args
parser.add_argument('--fineSize', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--max_bounces', type=int, default=2)
parser.add_argument('--distribution', type=str, default="profile")
# Output args
parser.add_argument('--num_imgs', type=int, default=1000)
parser.add_argument('--root_path', type=str, default="./datasets/renders/")
# Misc
parser.add_argument('--gpu_id', type=int, default=0, help="CUDA GPU id used for rendering. -1 for CPU")
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
now_string = "{}-{}_{}-{}".format(now.month, now.day, now.hour, now.minute)
subdir_name = "res{}mc{}_{}_{}".format(opt.fineSize, opt.num_samples, opt.distribution, now_string)
out_path = os.path.join(opt.root_path, subdir_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)
device = torch.device('cuda:{}'.format(opt.gpu_id) if opt.gpu_id != -1 else 'cpu')


# Render setup
texture = torch.tensor(imread(opt.texture_path), dtype=torch.float32, device=device)
render_logger = RenderLogger()
renderer = Render(
    opt.meshes_path, opt.envmaps_path, opt.fineSize, opt.num_samples, opt.max_bounces, device,
    channels = channels, logger = render_logger)

with open(os.path.join(out_path, "data.bak"), "a+") as backupfile:
    for i in range(opt.num_imgs):
        name = hashlib.sha256(str(random.randint(0, 100000000)).encode('utf-8')).hexdigest()[:6]
        print("Generating Image: #\t{} -- {}".format(i, name))
        out = renderer(texture, name)
        imwrite(out[:, :,   : 3], os.path.join(out_path, "img",      "{}.png".format(name)))
        imwrite(out[:, :,  3: 4], os.path.join(out_path, "mask",     "{}.png".format(name)))
        imwrite(out[:, :,  4: 7], os.path.join(out_path, "position", "{}.png".format(name)))
        imwrite(out[:, :,  7:10], os.path.join(out_path, "normal",   "{}.png".format(name)))
        imwrite(out[:, :, 10:13], os.path.join(out_path, "albedo",   "{}.png".format(name)))
        json.dump(render_logger.get_cfg(name), backupfile)

with open(os.path.join(out_path, "data.json"), "w") as metafile:
    json.dump(render_logger.get_all_cfgs(), metafile)
