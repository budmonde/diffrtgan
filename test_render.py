import torch

from util.render_util import MeshRenderLayer
from util.image_util import imread, imwrite

mesh_path = "datasets/meshes/octavia/octavia_uv.obj"
fineSize = 256
num_samples = 4
device = torch.device('cuda:0')
inp_path = "datasets/textures/checkerboard/train/checkerboard.png"

inp = torch.tensor(imread(inp_path), dtype=torch.float32, device=device)

render_layer = MeshRenderLayer(mesh_path, fineSize, num_samples, device)
out = render_layer(inp)
imwrite(out, "debug/test_render_out.png")
