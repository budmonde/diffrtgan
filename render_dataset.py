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
inp_path = "./datasets/textures/checkerboard/train/checkerboard.png"

out_path = "./datasets/renders/"
num_imgs = 1000

inp = torch.tensor(imread(inp_path), dtype=torch.float32, device=device)
render_logger = RenderLogger()
renderer = Render(
    meshes_path, envmaps_path, fineSize, num_samples, max_bounces, device, logger = render_logger)

for i in range(num_imgs):
    name = hashlib.sha256(str(random.randint(0, 100000000)).encode('utf-8')).hexdigest()[:6]
    print("Generating Image: #\t{} -- {}".format(i, name))
    out = renderer(inp, name)
    imwrite(out, "{}/img_out/{}.png".format(out_path, name))

with open("{}data.json".format(out_path), "w") as metafile:
    json.dump(render_logger.get_data(), metafile)
