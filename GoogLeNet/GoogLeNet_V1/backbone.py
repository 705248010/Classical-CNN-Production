import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Inception_block import Inception


"""
V1中所有Inception块的参数
3a: Inception(in_channels=[192, 192], out_channels=[64, 128, 32, 32])
3b: Inception(in_channels=[256, 256], out_channels=[128, 192, 96, 64])
4a: Inception(in_channels=[480, 480], out_channels=[192, 208, 48, 64])
4b: Inception(in_channels=[512, 512], out_channels=[160, 224, 64, 64])
4c: Inception(in_channels=[512, 512], out_channels=[128, 256, 64, 64])
4d: Inception(in_channels=[512, 512], out_channels=[112, 288, 64, 64])
4e: Inception(in_channels=[528, 528], out_channels=[256, 320, 128, 128])
5a: Inception(in_channels=[832, 832], out_channels=[256, 320, 128, 128])
5b: Inception(in_channels=[832, 832], out_channels=[384, 384, 128, 128])
"""


class GoogLeNetV1(nn.Module):
    """
    实现的是GoogLeNetV1，除了辅助分类器（原作者在v3的论文中提到没什么作用，所以就不实现了）都按照原始论文进行复现。
    """

    def __init__(self):
        super(GoogLeNetV1, self).__init__()
        self.net_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(7, 7), stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.LocalResponseNorm(size=56),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=56),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.net_2 = nn.Sequential(
            Inception(in_channels=[192, 192], out_channels=[64, 128, 32, 32]),
            Inception(in_channels=[256, 256], out_channels=[128, 192, 96, 64]),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )
        self.net_3 = nn.Sequential(
            Inception(in_channels=[480, 480], out_channels=[192, 208, 48, 64]),
            Inception(in_channels=[512, 512], out_channels=[160, 224, 64, 64]),
            Inception(in_channels=[512, 512], out_channels=[128, 256, 64, 64]),
            Inception(in_channels=[512, 512], out_channels=[112, 288, 64, 64]),
            Inception(in_channels=[528, 528],
                      out_channels=[256, 320, 128, 128]),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        )
        self.net_4 = nn.Sequential(
            Inception(in_channels=[832, 832],
                      out_channels=[256, 320, 128, 128]),
            Inception(in_channels=[832, 832],
                      out_channels=[384, 384, 128, 128]),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(p=0.4),
            nn.Flatten()
        )
        self.net_5 = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.Softmax()
        )
        self.backbone = nn.Sequential(
            self.net_1,
            self.net_2,
            self.net_3,
            self.net_4,
            self.net_5
        )
        self.backbone.apply(self._init_weights)

    def _init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        output = self.backbone(x)

        return output


a = GoogLeNetV1()
print(a)
# test = torch.rand((10, 3, 28, 28))
# b = a(test)
# print(b)
