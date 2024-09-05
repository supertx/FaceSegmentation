"""
@author supermantx
@date 2024/8/30 16:20
v2 predict segmentation mask on 224x224 image
v1 is a symmetric network that input with 112x112 and out predict mask straightly to 112x112
v2 is a asymmetric network that input with 224x224 and out predict mask is 56x56 and then upsample to 224x224
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from model.common import getActivation, getNorm


class DownSampleBlock(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UpSampleBlock(nn.Module):

    def __init__(self, in_channel, scale_factor=2, use_transpose=False):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, 3,
                               2, 1, 1) if use_transpose else nn.Upsample(scale_factor=scale_factor, mode="bilinear"),
            nn.ReLU if use_transpose else nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):

    def __init__(self, channel, norm="gn", num_groups=32):
        super().__init__()
        self.norm = getNorm(norm)(num_groups, channel)
        self.to_qkv = nn.Conv2d(channel, channel * 3, 1)
        self.to_out = nn.Conv2d(channel, channel, 1, 1)
        self.channel = channel

    def forward(self, x):
        b, c, w, h = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.channel, dim=1)
        q = q.permute(0, 2, 3, 1).reshape(b, w * h, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).reshape(b, w * h, c)

        attn = torch.bmm(q, k) * (c ** -0.5)

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).reshape(b, w, h, c).permute(0, 3, 1, 2)
        return self.to_out(out) + x


class SEAttentionBlock(nn.Module):
    """
    Self-Attention with Squeeze-and-Excitation
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x)
        out = torch.flatten(out, 1)
        out = self.attention(out).view(b, c, 1, 1)
        return x + out.expand_as(x)


class ResidualBlock(nn.Module):
    """
    two conv block with residual connection and closely follow a attention block
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 drop_out,
                 activation="relu",
                 norm="gn",
                 num_groups=32,
                 use_attention=True):
        super().__init__()
        self.activation = getActivation(activation)
        self.norm_1 = getNorm(norm)(num_groups, in_channel)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

        self.norm_2 = getNorm(norm)(num_groups, out_channel)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=drop_out), nn.Conv2d(out_channel, out_channel, 3, 1,
                                              1))
        self.shortcut = nn.Conv2d(
            in_channel, out_channel, 1,
            1) if in_channel != out_channel else nn.Identity()
        if use_attention:
            self.attn_blk = AttentionBlock(out_channel, norm, num_groups)

    def forward(self, x):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.shortcut(x)
        if hasattr(self, "attn_blk"):
            out = self.attn_blk(out)
        return out


class UNet(nn.Module):

    def __init__(self,
                 input_channel: int,
                 block_channel: List,
                 num_neck: int,
                 num_res_block: int = 2,
                 activation: str = "relu",
                 drop_out: float = 0.1,
                 norm: str = "gn",
                 num_groups: List = None):
        super().__init__()
        self.activation = getActivation(activation)
        self.stem = nn.Conv2d(input_channel, block_channel[0], 3, 1, 1)
        # downsample input 224x224 to 56x56
        self.head = nn.ModuleList()
        current_channel = block_channel[0]
        for i in range(2):
            self.head.append(
                ResidualBlock(current_channel,
                              block_channel[i + 1],
                              drop_out,
                              activation,
                              norm,
                              6,
                              use_attention=False))
            current_channel = block_channel[i + 1]
            self.head.append(DownSampleBlock(block_channel[i + 1]))

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = [block_channel[2]]
        for i, channel in enumerate(block_channel[3:]):
            for _ in range(num_res_block):
                self.downs.append(
                    ResidualBlock(current_channel, channel, drop_out,
                                  activation, norm, num_groups[i]))
                current_channel = channel
                channels.append(current_channel)
            if i != 2:
                self.downs.append(DownSampleBlock(current_channel))
                channels.append(current_channel)

        self.neck = nn.ModuleList()
        for _ in range(num_neck):
            self.neck.append(
                ResidualBlock(current_channel, current_channel, drop_out,
                              activation, norm, num_groups[-1]))

        num_groups = list(reversed(list(map(
            lambda x: x * 2, num_groups))))[:-1] + [num_groups[0]]
        for i, channel in enumerate(list(reversed(block_channel[3:]))):
            current_channel = channel
            for j in range(num_res_block + 1):
                if j == 1:
                    channel = channel // 2
                self.ups.append(
                    ResidualBlock(channels.pop() + current_channel, channel,
                                  drop_out, activation, norm, num_groups[i]))
                current_channel = channel

            if i != 2:
                self.ups.append(UpSampleBlock(current_channel))
        # 72x56x56 -> 18x56x56
        self.tail = nn.Sequential(
            ResidualBlock(block_channel[2], block_channel[0], drop_out, activation, norm, num_groups[-1] // 2, False),
            UpSampleBlock(block_channel[0], 4, False))

    def forward(self, x):
        out = self.stem(x)
        for block in self.head:
            out = block(out)

        down_outs = [out]

        for block in self.downs:
            out = block(out)
            down_outs.append(out)

        for block in self.neck:
            out = block(out)

        for block in self.ups:
            if isinstance(block, ResidualBlock):
                out = torch.cat([out, down_outs.pop()], dim=1)
            out = block(out)
        out = self.tail(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = UNet(3,
                 [18, 36, 72, 144, 288, 576],
                 2, 2,
                 num_groups=[12, 12, 24])
    summary(model, (3, 224, 224), device="cpu")
