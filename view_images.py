import os

IMGS_DIR = './datasets/poses/octavia_clean/100_10/'

with open('./view_poses.html', 'w') as f:
    fn_list = sorted(os.listdir(IMGS_DIR))
    for fn in fn_list:
        impath = os.path.join(IMGS_DIR, fn)
        img_str = '<img src="{}" />'.format(impath)
        f.write(img_str)
        label_str = '<label>{}</label>'.format(fn)
        f.write(label_str)
