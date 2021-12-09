import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Inception(nn.Module):
    """
    in_channels: list. Example, [1x1, max_pooling]
    out_channels: list. Example, [1x1, 3x3, 5x5, max_pooling]
    """

    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(1, 1), stride=1)
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[2], kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
        )
        self.block_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[3], kernel_size=(1, 1), stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        a = F.relu(self.conv_1x1(x))
        b = self.block_1(x)
        c = self.block_2(x)
        d = self.block_3(x)
        output = torch.cat([a, b, c, d], dim=1)

        return output

# a = Inception(in_channels=[192, 192], out_channels=[64, 128, 32, 32])
# test = torch.rand((1, 192, 56, 56))
# b = a(test)
# print(b)
