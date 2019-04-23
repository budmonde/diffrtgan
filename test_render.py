import torch

from util.image_util import imread, imwrite
from util.render_util import RenderConfig
from util.torch_util import RenderLayer


meshes_path = "./datasets/meshes/serialized"
envmaps_path = "./datasets/envmaps/hdrs"
fineSize = 512
num_samples = 200
max_bounces = 1
device = torch.device('cuda:0')
opaque_path = "./datasets/textures/checkerboard.png"
alpha_path = "./datasets/textures/decal.png"

render_config = RenderConfig({
    'test': {
        'geo_mesh_path': './datasets/meshes/serialized/octavia_clean.pth',
        'tex_envmap_path': './datasets/envmaps/hdrs/lenong_1_1k.hdr',
        'tex_envmap_signal_mean': 0.5,
        'tex_envmap_rangle': 0.0,
        'geo_rotation': [2.4363371599267785, 0.14922565104551516, 0.0],
        'geo_translation': [0.0, -0.75, 0.0],
        'geo_distance': 7.0,
        'render_seed': 0,
    }
})
render_config.set_config('test')
render_kwargs = {
    "meshes_path":  meshes_path,
    "envmaps_path": envmaps_path,
    "out_sz":       fineSize,
    "num_samples":  num_samples,
    "max_bounces":  max_bounces,
    "device":       device,
    "channels":     ["radiance"],
    "config":       render_config,
}
render_layer = RenderLayer(**render_kwargs)

opaque = torch.tensor(imread(opaque_path), dtype=torch.float32, device=device)
alpha = torch.tensor(imread(alpha_path), dtype=torch.float32, device=device)

out = render_layer(opaque)
imwrite(out, "debug/test_render_opaque_out.png")
out = render_layer(alpha)
imwrite(out, "debug/test_render_alpha_out.png")
