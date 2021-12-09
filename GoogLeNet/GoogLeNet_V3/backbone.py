import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from inception_block import *


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV3, self).__init__()
        self.stage = stage

        self.block_1 = nn.Sequential(
            BasicConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            BasicConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            BasicConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block_2 = nn.Sequential(
            BasicConvBlock(in_channels=64, out_channels=80, kernel_size=3, stride=1),
            BasicConvBlock(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.block_3 = nn.Sequential(
            InceptionBlockA(in_channels=192, out_channels=[64, 48, 64, 64, 96, 32]),
            InceptionBlockA(in_channels=256, out_channels=[64, 48, 64, 64, 96, 64]),
            InceptionBlockA(in_channels=288, out_channels=[64, 48, 64, 64, 96, 64])
        )

        self.block_4 = nn.Sequential(
            InceptionBlockD(in_channels=288, out_channels=[384, 384, 64, 96]),
            InceptionBlockB(in_channels=768, out_channels=[192, 128, 192, 128, 192, 192]),
            InceptionBlockB(in_channels=768, out_channels=[192, 160, 192, 160, 192, 192]),
            InceptionBlockB(in_channels=768, out_channels=[192, 160, 192, 160, 192, 192]),
            InceptionBlockB(in_channels=768, out_channels=[192, 192, 192, 192, 192, 192]),
        )

        if self.stage=='train':
            self.aux_logits = InceptionAux(in_channels=768, out_channels=num_classes)

        self.block_5 = nn.Sequential(
            InceptionBlockE(in_channels=768, out_channels=[192, 320, 192, 192]),
            InceptionBlockC(in_channels=1280, out_channels=[320, 384, 384, 448, 384, 192]),
            InceptionBlockC(in_channels=2048, out_channels=[320, 384, 384, 448, 384, 192]),
        )

        self.max_pooling = nn.MaxPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        aux = x = self.block_4(x)
        x = self.block_5(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)

        if self.stage == 'train':
            aux = self.aux_logits(aux)
            return aux, out
        else:
            return out

model = InceptionV3()
print(model)

input = torch.randn(1, 3, 299, 299)
aux, out = model(input)
print(aux.shape)
print(out.shape)