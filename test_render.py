import torch

from util.image_util import imread, imwrite
from util.torch_util import RenderLayer


meshes_path = "./datasets/meshes/one_mtl"
envmaps_path = "./datasets/envmaps/one"
fineSize = 256
num_samples = 4
max_bounces = 2
device = torch.device('cuda:0')
opaque_path = "./datasets/textures/checkerboard/train/checkerboard.png"
alpha_path = "./datasets/textures/decal/train/decal.png"

opaque = torch.tensor(imread(opaque_path), dtype=torch.float32, device=device)
alpha = torch.tensor(imread(alpha_path), dtype=torch.float32, device=device)

render_layer = RenderLayer(meshes_path, envmaps_path, fineSize, num_samples, max_bounces, device)

out = render_layer(opaque)
imwrite(out, "debug/test_render_opaque_out.png")
out = render_layer(alpha)
imwrite(out, "debug/test_render_alpha_out.png")
