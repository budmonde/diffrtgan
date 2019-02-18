import torch.nn as nn

from .render_util import *
from .transform_util import *


class RenderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RenderLayer, self).__init__()
        self.renderer = Render(*args, **kwargs)

    def forward(self, input):
        out = self.renderer(input)
        return out

class CompositLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CompositLayer, self).__init__()
        self.transform = Composit(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)


class HWC2CHWLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HWC2CHWLayer, self).__init__()
        self.transform = HWC2CHW(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class CHW2HWCLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CHW2HWCLayer, self).__init__()
        self.transform = CHW2HWC(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class NormalizeLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NormalizeLayer, self).__init__()
        self.transform = Normalize(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class StripBatchDimLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StripBatchDimLayer, self).__init__()
        self.transform = StripBatchDim(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class AddBatchDimLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AddBatchDimLayer, self).__init__()
        self.transform = AddBatchDim(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class NormalizedCompositLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NormalizedCompositLayer, self).__init__()
        self.model = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
                CompositLayer(*args, **kwargs),
                HWC2CHWLayer(),
                NormalizeLayer(0.5, 0.5),
                AddBatchDimLayer(),
        )

    def forward(self, input):
        out = self.model.forward(input)
        return out

class NormalizedRenderLayer(nn.Module):
    # TODO: Make this argument passing a little more clean
    def __init__(self, mesh_path, bkgd, out_sz, num_samples, max_bounces, device, config = None):
        super(NormalizedRenderLayer, self).__init__()
        self.model = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
                CompositLayer(bkgd, out_sz, device),
                RenderLayer(mesh_path, out_sz, num_samples, max_bounces, device, config = config),
                HWC2CHWLayer(),
                NormalizeLayer(0.5, 0.5),
                AddBatchDimLayer(),
        )

    def forward(self, input):
        out = self.model.forward(input)
        return out