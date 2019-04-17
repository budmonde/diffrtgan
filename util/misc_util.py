import hashlib
import os
import random

# Hashname generator
def gen_hash(length):
    return hashlib.sha256(str(random.randint(0, 1e10))\
                  .encode('utf-8')).hexdigest()[:length]

# List of Child paths
def get_child_paths(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def get_fn(path, ext=False):
    fn = path.split('/')[-1]
    if not ext:
        fn = fn.split('.')[0]
    return fn

def get_ext(path):
    return path.split('.')[-1]
