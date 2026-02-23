import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeViewConv2d(nn.Conv2d):
    def __init__(self, custom_padding, *args, **kwargs):
        if custom_padding:
            raise NotImplementedError()
        super(RangeViewConv2d, self).__init__(*args, **kwargs)