import argparse
import os

from util.misc_util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    opt = parser.parse_args()

    name = get_fn(opt.path)

    with open(f'./views/view_{name}.html', 'w') as f:
        fn_list = sorted(os.listdir(opt.path))
        for fn in fn_list:
            impath = os.path.join(opt.path, fn)
            img_str = '<img src="../{}" />'.format(impath)
            f.write(img_str)
            label_str = '<label>{}</label>'.format(fn)
            f.write(label_str)

if __name__ == '__main__':
    main()
