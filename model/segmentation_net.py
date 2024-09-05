"""
@author supermantx
@date 2024/8/29 16:48
"""
from typing import List

from torch import nn
from torch.nn import BCELoss
from torch.nn import functional as F
# from model.unet_v1 import UNet
from model.unet_v2 import UNet


class SegmentationNet(nn.Module):

    def __init__(self,
                 input_channel: int,
                 block_channel: List,
                 num_neck: int,
                 num_res_block: int = 2,
                 activation: str = "relu",
                 drop_out: float = 0.1,
                 norm: str = "gn",
                 num_groups: List = None,
                 class_num: int = 18
                 ):
        super().__init__()
        self.unet = UNet(input_channel, block_channel, num_neck, num_res_block, activation, drop_out, norm, num_groups)
        self.classifier = nn.Conv2d(block_channel[0], class_num, 3, 1, 1)

        self.loss = BCELoss()
        self.training = True

    def forward(self, x, mask=None):
        out = self.unet(x)
        out = self.classifier(out)
        out = F.sigmoid(out)
        if self.training:
            loss = self.loss(out, mask)
            return loss
        else:
            return out
