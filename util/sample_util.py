import math
import random

from util.misc_util import *


class ConfigSampler(object):
    def __init__(self, samplers):
        super(ConfigSampler, self).__init__()
        self.samplers = samplers

    def generate(self):
        config = dict()
        for key, sampler in self.samplers.items():
            config[key] = sampler()
        return config

def PathSamplerFactory(root_dir, ext=None):
    path_list = get_child_paths(root_dir, ext)
    def sampler():
        return random.choice(path_list)
    return sampler

def ConstantSamplerFactory(value):
    def sampler():
        return value
    return sampler

def RandIntSamplerFactory(min_val, max_val):
    def sampler():
        return random.randint(min_val, max_val)
    return sampler

def UniformSamplerFactory(min_val, max_val):
    def sampler():
        return random.uniform(min_val, max_val)
    return sampler

def RGBFileSamplerFactory(fpath):
    with open(fpath) as f:
        options = [
                [float(v)/255.0 for v in s.split(' ')]
                for s in f.read().strip('\n').split('\n')]
    def sampler():
        return random.choice(options)
    return sampler

# Note: the ranges are normalized to the range (0, 1)
def HemisphereSamplerFactory(azimuth_range, elevation_range, up_range):
    def sampler():
        azim = random.uniform(*azimuth_range) * math.pi * 2
        elev = math.acos(1 - random.uniform(*elevation_range))
        ornt = random.uniform(*up_range)
        return [azim, elev, ornt]
    return sampler
def BoxSamplerFactory(x_range, y_range, z_range):
    def sampler():
        x, y, z = random.uniform(*x_range),\
                  random.uniform(*y_range),\
                  random.uniform(*z_range)
        return [x, y, z]
    return sampler
