import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from torch.autograd import Variable
from .utils import conv, deconv, update_registered_buffers
from compressai.layers import GDN, MaskedConv2d, ResidualBlock, AttentionBlock, ResidualBlockWithStride, ResidualBlockUpsample


def maxpool2d():
    return nn.MaxPool2d(2)

class Saliency(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encode = nn.Sequential(
            conv(3, 64, stride=1),
            ResidualBlockWithStride(64, 64, stride=2),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlockWithStride(128, 128, stride=2),
            ResidualBlock(128, 512),
            ResidualBlock(512, 512),
            ResidualBlockWithStride(512, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlockWithStride(512, 512, stride=2),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(512, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 512, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.decode = nn.Sequential(
            deconv(512, 512, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            deconv(512, 512, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            deconv(512, 512, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            deconv(512, 512, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            deconv(512, 512, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            deconv(512, 512, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            deconv(512, 256, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            deconv(256, 256, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            deconv(256, 256, kernel_size=3, stride=1),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            deconv(256, 128, stride=1),
            # nn.ReLU(inplace=True),
            deconv(128, 128, stride=1),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            deconv(128, 64, stride=1),
            # nn.ReLU(inplace=True),
            deconv(64, 64, stride=1),
            # nn.ReLU(inplace=True)
        )

        self.generate = nn.Sequential(
            deconv(64, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.encode(x)
        y = self.avg_pool(x)
        y = self.conv_du(y)
        x = x * y
        x = self.decode(x)
        x = self.generate(x)

        return x