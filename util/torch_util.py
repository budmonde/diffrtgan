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

class PostCompositLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PostCompositLayer, self).__init__()
        self.transform = PostComposit(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class CompositLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CompositLayer, self).__init__()
        self.transform = Composit(*args, **kwargs)

    def forward(self, input):
        return self.transform(input)

class GaussianNoiseLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GaussianNoiseLayer, self).__init__()
        self.transform = GaussianNoise(*args, **kwargs)

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
    def __init__(self, render_kwargs, composit_kwargs):
        super(NormalizedRenderLayer, self).__init__()
        self.model = nn.Sequential(
                StripBatchDimLayer(),
                NormalizeLayer(-1.0, 2.0),
                CHW2HWCLayer(),
                RenderLayer(**render_kwargs),
                CompositLayer(**composit_kwargs),
                HWC2CHWLayer(),
                NormalizeLayer(0.5, 0.5),
                AddBatchDimLayer(),
        )

    def forward(self, input):
        out = self.model.forward(input)
        return out
