"""
@author supermantx
@date 2024/8/28 11:36
"""
from torch import nn
from torch.nn import functional as F

__all__ = ["getActivation", "getNorm"]

def getActivation(activation):
    if activation.lower() == "relu":
        return F.relu
    elif activation.lower() == "gelu":
        return F.gelu
    elif activation.lower() == "lrelu":
        return F.leaky_relu
    elif activation.lower() == "tanh":
        return F.tanh
    else:
        raise ValueError("activation function not supported")


def getNorm(norm):
    if norm.lower() == "bn":
        return nn.BatchNorm2d
    elif norm.lower() == "gn":
        # num_channels must be divisible by num_groups
        return nn.GroupNorm
    elif norm.lower() == "in":
        return nn.InstanceNorm2d
    else:
        raise ValueError("normalization function not supported")
