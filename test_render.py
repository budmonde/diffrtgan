import torch

from util.image_util import imread, imwrite
from util.torch_util import RenderLayer


meshes_path = "./datasets/meshes/one"
envmaps_path = "./datasets/envmaps/rasters"
fineSize = 256
num_samples = 4
max_bounces = 2
device = torch.device('cuda:0')
inp_path = "./datasets/textures/checkerboard/train/checkerboard.png"

inp = torch.tensor(imread(inp_path), dtype=torch.float32, device=device)

render_layer = RenderLayer(meshes_path, envmaps_path, fineSize, num_samples, max_bounces, device)
out = render_layer(inp)
imwrite(out, "debug/test_render_out.png")
