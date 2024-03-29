import numpy as np
import matplotlib
import OpenEXR
import Imath
import imageio
import skimage
import skimage.io
from skimage.transform import resize
import torch
import os

def imwrite(img, filename, normalize = False):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if isinstance(img, torch.Tensor):
        img = img.cpu().data.numpy()

    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng

    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))

    if filename[-4:] == '.exr':
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        pixels_r = img_r.astype(np.float16).tostring()
        pixels_g = img_g.astype(np.float16).tostring()
        pixels_b = img_b.astype(np.float16).tostring()
        HEADER = OpenEXR.Header(img.shape[1], img.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
        exr = OpenEXR.OutputFile(filename, HEADER)
        exr.writePixels({'R': pixels_r, 'G': pixels_g, 'B': pixels_b})
        exr.close()
    else:
        img = np.power(np.clip(img, 0.0, 1.0), 1.0/2.2)
        img = (img * 255.0).astype('uint8')
        skimage.io.imsave(filename, img)

def imread(filename):
    if (filename[-4:] == '.exr'):
        file = OpenEXR.InputFile(filename)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        redstr = file.channel('R', pt)
        red = np.fromstring(redstr, dtype = np.float32)
        red.shape = (size[1], size[0])
        greenstr = file.channel('G', pt)
        green = np.fromstring(greenstr, dtype = np.float32)
        green.shape = (size[1], size[0])
        bluestr = file.channel('B', pt)
        blue = np.fromstring(bluestr, dtype = np.float32)
        blue.shape = (size[1], size[0])
        return np.stack([red, green, blue], axis=-1).astype(np.float32)
    elif (filename[-4:] == '.hdr'):
        im = imageio.imread(filename)
    else:
        im = skimage.io.imread(filename)

    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
        return np.power(skimage.img_as_float(im).astype(np.float32), 2.2)
    elif im.shape[2] == 4:
        alpha = (im[:, :, 3] / 255.).astype(np.float32)
        im = im[:, :, :3]
        im = np.power(skimage.img_as_float(im).astype(np.float32), 2.2)
        return np.dstack([im, alpha])
    else:
        return np.power(skimage.img_as_float(im).astype(np.float32), 2.2)

# Not the best place for this function
def grey2heatmap(image_torch, size):
    im = np.array(image_torch[0,...].cpu()).transpose([1, 2, 0])

    max_val = max(abs(np.max(im)), abs(np.min(im)))

    im = (im / (2 * max_val)) + 0.5

    im = resize(im, (size, size))

    rgb = matplotlib.cm.get_cmap('viridis')(im[...,0])[...,:3]

    rgb = (rgb - 0.5) / 0.5

    rgb_torch = torch.tensor(rgb.transpose([2, 0, 1]), dtype=torch.float32).unsqueeze(0);

    return rgb_torch
