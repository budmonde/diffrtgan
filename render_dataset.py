import hashlib
import random
import json

import torch

from util.image_util import imread, imwrite
from util.render_util import RenderLogger, Render


meshes_path = "./datasets/meshes/one_mtl"
envmaps_path = "./datasets/envmaps/rasters"
fineSize = 512
num_samples = 20
max_bounces = 2
device = torch.device('cuda:0')
channels = [
    'radiance',
    'alpha',
    'position',
    'shading_normal',
    'diffuse_reflectance'
]

inp_path = "./datasets/textures/checkerboard/train/checkerboard.png"

out_path = "./datasets/renders/"
num_imgs = 1000

inp = torch.tensor(imread(inp_path), dtype=torch.float32, device=device)
render_logger = RenderLogger()
renderer = Render(
    meshes_path, envmaps_path, fineSize, num_samples, max_bounces, device,
    channels = channels, logger = render_logger)

for i in range(num_imgs):
    name = hashlib.sha256(str(random.randint(0, 100000000)).encode('utf-8')).hexdigest()[:6]
    print("Generating Image: #\t{} -- {}".format(i, name))
    out = renderer(inp, name)
    imwrite(out[:, :, :3], "{}/img/{}.png".format(out_path, name))
    imwrite(out[:, :, 3:4], "{}/mask/{}.png".format(out_path, name))
    imwrite(out[:, :, 4:7], "{}/position/{}.png".format(out_path, name))
    imwrite(out[:, :, 7:10], "{}/normal/{}.png".format(out_path, name))
    imwrite(out[:, :, 10:13], "{}/albedo/{}.png".format(out_path, name))

with open("{}data.json".format(out_path), "w") as metafile:
    json.dump(render_logger.get_data(), metafile)
