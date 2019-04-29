import os

import numpy as np
import skimage.filters
from skimage.transform import resize

from util.image_util import imread, imwrite
from util.misc_util import *

def height_dirt(position, normal, mask):
    strength = 0.1
    ramp = 4.0

    color = position[...,1:2]**ramp * 1 / strength
    color = np.clip(color, 0.0, 1.0)
    color = np.repeat(color, 3, axis=-1)
    color = color * mask

    alpha = 1 - color[...,:1]
    alpha = (1 - mask[...,:1]) + alpha * mask[...,:1]

    return np.concatenate([color, alpha], axis=-1)

def curve_dirt(position, normal, mask):

    # Normalize normal map
    n = (normal - 0.5) / 0.5
    n = n / np.sqrt(np.sum(n * n, axis=-1, keepdims=True))

    dn_u = np.sum(np.roll(n, 1, axis=1) - n, axis=-1, keepdims=True)
    dn_v = np.sum(np.roll(n, 1, axis=0) - n, axis=-1, keepdims=True)

    dn = dn_u + dn_v

    H = np.sqrt(np.sum(dn * dn, axis=-1))
    H = (H - np.min(H)) / (np.max(H) - np.min(H))

    H = np.clip((H*200)**10, 0, 1)
    H = skimage.filters.gaussian(H, sigma = 1)
    H = skimage.filters.unsharp_mask(H)
    H[H<0.7] = 0

    color = np.zeros((256, 256, 3))
    color = color * mask

    alpha = H[...,np.newaxis]
    alpha = (1 - mask[...,:1]) + alpha * mask[...,:1]

    return np.concatenate([color, alpha], axis=-1)

def composit(img, bkgd):
    return img[...,:3] * img[...,-1:] + bkgd * (1 - img[...,-1:])

def main():
    MESH_ROOT = './datasets/meshes/clean_serialized'
    CAR_ROSTER = [get_fn(path) for path in get_child_paths(MESH_ROOT)]

    POS_GBUF_ROOT = './datasets/gbuffers/position'
    NOR_GBUF_ROOT = './datasets/gbuffers/normal'
    MSK_GBUF_ROOT = './datasets/gbuffers/mask'

    HEIGHT_DIRT_ROOT = './datasets/textures/height'
    CURVE_DIRT_ROOT = './datasets/textures/curve'

    for name in CAR_ROSTER:
        print(f'Generating dirt for {name}')
        pos = imread(os.path.join(POS_GBUF_ROOT, f'{name}.png'))
        nrm = imread(os.path.join(NOR_GBUF_ROOT, f'{name}.png'))
        msk = imread(os.path.join(MSK_GBUF_ROOT, f'{name}.png'))

        h_dirt = height_dirt(pos, nrm, msk)
        imwrite(h_dirt, os.path.join(HEIGHT_DIRT_ROOT, f'{name}.png'))
        c_dirt = curve_dirt(pos, nrm, msk)
        imwrite(c_dirt, os.path.join(CURVE_DIRT_ROOT, f'{name}.png'))

if __name__ == '__main__':
    main()
