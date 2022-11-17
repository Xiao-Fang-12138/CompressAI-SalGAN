import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from torch.autograd import Variable
from .utils import conv, deconv, update_registered_buffers
from compressai.layers import GDN, MaskedConv2d, ResidualBlock


def maxpool2d():
    return nn.MaxPool2d(2)

class Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encode = nn.Sequential(
            conv(3, 64, stride=1),
            nn.ReLU(inplace=True),
            conv(64, 64, stride=1),
            nn.ReLU(inplace=True),
            maxpool2d(),
            conv(64, 128, stride=1),
            nn.ReLU(inplace=True),
            conv(128, 128, stride=1),
            nn.ReLU(inplace=True),
            maxpool2d(),
            conv(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            maxpool2d(),
            conv(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            maxpool2d(),
            conv(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
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
        x = self.decode(x)
        x = self.generate(x)

        return x