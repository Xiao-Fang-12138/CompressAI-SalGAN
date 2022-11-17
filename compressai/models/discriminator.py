import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from .utils import conv, deconv



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.analysis = nn.Sequential(  # [-1, 4, 256,192]
            conv(4, 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            conv(3, 32, kernel_size=3, stride=1),  # [-1, 32, 256, 192]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            conv(32, 64, kernel_size=3, stride=1),  # [-1, 64, 128, 96]
            nn.ReLU(inplace=True),
            conv(64, 64, kernel_size=3, stride=1),  # [-1, 64, 128, 96]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [-1,64,64,48]
            conv(64, 64, kernel_size=3, stride=1),  # [-1,64,64,48]
            nn.ReLU(inplace=True),
            conv(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # [-1,64,32,24]
        )
        self.detect = nn.Sequential(
            nn.Linear(64 * 32 * 24, 100),
            nn.Tanh(),
            nn.Linear(100, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.analysis(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.detect(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # print(m.weight.data.shape)
                # print('old conv layer!')
                # print(m.weight.data.min())
                # print(m.weight.data.max())
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # print('new conv layer!')
                # print(m.weight.data.min())
                # print(m.weight.data.max())
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
