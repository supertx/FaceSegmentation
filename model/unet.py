"""
@author supermantx
@date 2024/8/28 11:17
"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from calflops import calculate_flops

from model.common import getActivation, getNorm


class DownSampleBlock(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UpSampleBlock(nn.Module):

    def __init__(self, in_channel, use_transpose=False):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, 3,
                               2, 1, 1) if use_transpose else nn.Upsample(scale_factor=2, mode="nearest"),
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
                 ):
        super().__init__()
        self.activation = getActivation(activation)
        self.norm_1 = getNorm(norm)(num_groups, in_channel)
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

        self.norm_2 = getNorm(norm)(num_groups, out_channel)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1, 1) if in_channel != out_channel else nn.Identity()
        self.attn_blk = AttentionBlock(out_channel, norm, num_groups)

    def forward(self, x):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.shortcut(x)
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
                 num_groups: List = None
                 ):
        super().__init__()
        self.activation = getActivation(activation)
        self.stem = nn.Conv2d(input_channel, block_channel[0], 3, 1, 1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        current_channel = block_channel[0]
        channels = [block_channel[0]]
        for i, channel in enumerate(block_channel[1:]):
            for _ in range(num_res_block):
                self.downs.append(ResidualBlock(
                    current_channel,
                    channel,
                    drop_out,
                    activation,
                    norm,
                    num_groups[i]
                ))
                current_channel = channel
                channels.append(current_channel)
            if i != len(block_channel) - 2:
                self.downs.append(DownSampleBlock(current_channel))
                channels.append(current_channel)

        self.neck = nn.ModuleList()
        for _ in range(num_neck):
            self.neck.append(ResidualBlock(
                current_channel,
                current_channel,
                drop_out,
                activation,
                norm,
                num_groups[-1]
            ))

        num_groups = list(reversed(list(map(lambda x: x * 2, num_groups))))[:-1] + [num_groups[0]]
        for i, channel in enumerate(list(reversed(block_channel[1:]))):
            current_channel = channel
            for j in range(num_res_block + 1):
                if j == 1:
                    channel = channel // 2
                self.ups.append(ResidualBlock(
                    channels.pop() + current_channel,
                    channel,
                    drop_out,
                    activation,
                    norm,
                    num_groups[i]
                ))
                current_channel = channel

            if i != len(block_channel) - 2:
                self.ups.append(UpSampleBlock(current_channel, False))

    def forward(self, x):
        out = self.stem(x)
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
        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = UNet(3,
                 [18, 36, 72, 144, 288],
                 2, 2,
                 num_groups=[6, 6, 12, 24])
    model.cuda()
    batch_size = 4
    input_shape = (batch_size, 3, 112, 112)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)