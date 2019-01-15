import torch

from util.render_util import CompositLayer, CornerCrop, MeshRenderLayer
from util.image_util import imread, imwrite

fineSize = 500
device = torch.device('cuda:0')
bkgd_path = "datasets/textures/transparency/transparency.png"
inp_path = "datasets/textures/transparent_checkerboard/train/transparent_checkerboard.png"

crop_fn = CornerCrop(fineSize)
#bkgd = torch.tensor((0.7, 0.0, 0.7))\
#            .expand(fineSize, fineSize, 3)
bkgd = torch.tensor(
        crop_fn(imread(bkgd_path)), dtype=torch.float32, device=device)
inp = torch.tensor(
        crop_fn(imread(inp_path)), dtype=torch.float32, device=device)

composit_layer = CompositLayer(bkgd, fineSize, device)
composit = composit_layer(inp)
imwrite(composit, "debug/test_composit_out.png")

mesh_path = "datasets/meshes/octavia/octavia_uv.obj"
fineSize = 256
num_samples = 4

render_layer = MeshRenderLayer(mesh_path, fineSize, num_samples, device)
out = render_layer(composit)
imwrite(out, "debug/test_render_composit_out.png")
