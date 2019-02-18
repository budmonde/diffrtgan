import torch

from util.image_util import imread, imwrite
from util.torch_util import CompositLayer, RenderLayer
from util.transform_util import CornerCrop, Resize


fineSize = 500
device = torch.device('cuda:0')
bkgd_path = "./datasets/textures/transparency/transparency.png"
inp_path = "./datasets/textures/transparent_checkerboard/train/transparent_checkerboard.png"

#transform = CornerCrop(fineSize)
transform = Resize(fineSize)

#bkgd = torch.tensor((0.7, 0.0, 0.7))\
#            .expand(fineSize, fineSize, 3)
#bkgd = torch.tensor(
#        crop_fn(imread(bkgd_path)), dtype=torch.float32, device=device)
bkgd = None
inp = torch.tensor(transform(imread(inp_path)), dtype=torch.float32, device=device)

composit_layer = CompositLayer(bkgd, fineSize, device)
composit = composit_layer(inp)
imwrite(composit, "debug/test_transform_out.png")
