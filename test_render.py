import torch

from util.image_util import imread, imwrite
from util.render_util import RenderConfig
from util.torch_util import RenderLayer


meshes_path = "./datasets/meshes/one"
envmaps_path = "./datasets/envmaps/one"
fineSize = 512
num_samples = 200
max_bounces = 1
device = torch.device('cuda:0')
opaque_path = "./datasets/textures/checkerboard.png"
alpha_path = "./datasets/textures/decal.png"

render_config = RenderConfig({
    'test': {
        'geo_mesh_path': './datasets/meshes/one/octavia_clean.obj',
        'tex_envmap_path': './datasets/envmaps/one/sunsky.exr',
        'geo_rotation': [1.5707963267948966, 0.15, 0.0],
        'geo_translation': [0.0, -0.65, 0.0],
        'geo_distance': 8.3,
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
    "config":       render_config,
}
render_layer = RenderLayer(**render_kwargs)

opaque = torch.tensor(imread(opaque_path), dtype=torch.float32, device=device)
alpha = torch.tensor(imread(alpha_path), dtype=torch.float32, device=device)

out = render_layer(opaque)
imwrite(out, "debug/test_render_opaque_out.png")
out = render_layer(alpha)
imwrite(out, "debug/test_render_alpha_out.png")
