import os
import numpy as np

from PIL import Image
from image_util import imwrite

ROOT_DIR = './datasets/cosy/'
LABEL_DIR = '{}label/'.format(ROOT_DIR)
MASK_DIR = '{}mask/'.format(ROOT_DIR)

if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)

fn_list = os.listdir(LABEL_DIR)
for fn in fn_list:
    fpath = os.path.join(LABEL_DIR, fn)
    label = np.array(Image.open(fpath))
    label[label != 250] = 0.
    label[label == 250] = 1.
    imwrite(label, os.path.join(MASK_DIR, fn))
