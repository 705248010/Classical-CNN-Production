import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        temp = self.conv(x)
        temp = self.bn(temp)
        return F.relu(temp, inplace=True)


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockA, self).__init__()
        """
        in_channels: int, input_channels
        out_channels: list, [out_channels_1, out_channels_2_reduce, out_channels_2, out_channels_3_reduce, out_channels_3, out_channels_4]
        """
        self.branch_1 = BasicConvBlock(
            in_channels=in_channels, out_channels=out_channels[0], kernel_size=1)

        self.branch_2 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[1], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=5, padding=2),
        )

        self.branch_3 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[3], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=3, padding=1),
            BasicConvBlock(
                in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=3, padding=1),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[5], kernel_size=1),
        )

    def forward(self, x):
        b_1 = self.branch_1(x)
        b_2 = self.branch_2(x)
        b_3 = self.branch_3(x)
        b_4 = self.branch_4(x)
        out = torch.cat([b_1, b_2, b_3, b_4], dim=1)
        return out


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockB, self).__init__()
        """
        in_channels: int, input_channels
        out_channels: list, [out_channels_1, out_channels_2_reduce, out_channels_2, out_channels_3_reduce, out_channels_3, out_channels_4]
        """
        self.branch_1 = BasicConvBlock(
            in_channels=in_channels, out_channels=out_channels[0], kernel_size=1)

        self.branch_2 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[1], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=[1, 7], padding=[0, 3]),
            BasicConvBlock(
                in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=[7, 1], padding=[3, 0]),
        )

        self.branch_3 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[3], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=[1, 7], padding=[0, 3]),
            BasicConvBlock(
                in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=[7, 1], padding=[3, 0]),
            BasicConvBlock(
                in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=[1, 7], padding=[0, 3]),
            BasicConvBlock(
                in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=[7, 1], padding=[3, 0]),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[5], kernel_size=1),
        )

    def forward(self, x):
        b_1 = self.branch_1(x)
        b_2 = self.branch_2(x)
        b_3 = self.branch_3(x)
        b_4 = self.branch_4(x)
        out = torch.cat([b_1, b_2, b_3, b_4], dim=1)
        return out


class InceptionBlockC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockC, self).__init__()
        """
        in_channels: int, input_channels
        out_channels: list, [out_channels_1, out_channels_2_reduce, out_channels_2, out_channels_3_reduce, out_channels_3, out_channels_4]
        """
        self.branch_1 = BasicConvBlock(
            in_channels=in_channels, out_channels=out_channels[0], kernel_size=1)

        self.branch_2_conv_1 = BasicConvBlock(
            in_channels, out_channels=out_channels[1], kernel_size=1)
        self.branch_2_conv_2a = BasicConvBlock(
            out_channels[1], out_channels=out_channels[2], kernel_size=[1, 3], padding=[0, 1])
        self.branch_2_conv_2b = BasicConvBlock(
            out_channels[1], out_channels=out_channels[2], kernel_size=[3, 1], padding=[1, 0])

        self.branch_3_conv_1 = BasicConvBlock(
            in_channels, out_channels=out_channels[3], kernel_size=1)
        self.branch_3_conv_2 = BasicConvBlock(
            in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=3, stride=1, padding=1)
        self.branch_3_conv_3a = BasicConvBlock(
            in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=[3, 1], padding=[1, 0])
        self.branch_3_conv_3b = BasicConvBlock(
            in_channels=out_channels[4], out_channels=out_channels[4], kernel_size=[1, 3], padding=[0, 1])

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[5], kernel_size=1),
        )

    def forward(self, x):
        b_1 = self.branch_1(x)
        b_2_conv = self.branch_2_conv_1(x)
        b_2 = torch.cat([self.branch_2_conv_2a(b_2_conv),
                         self.branch_2_conv_2b(b_2_conv)], dim=1)
        b_3_conv = self.branch_3_conv_2(self.branch_3_conv_1(x))
        b_3 = torch.cat([self.branch_3_conv_3a(b_3_conv),
                         self.branch_3_conv_3b(b_3_conv)], dim=1)
        b_4 = self.branch_4(x)
        out = torch.cat([b_1, b_2, b_3, b_4], dim=1)
        return out


class InceptionBlockD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockD, self).__init__()
        """
        in_channels: int, input_channels
        out_channels: list, [out_channels1reduce, out_channels1, out_channels2reduce, out_channels2]
        """
        self.branch_1 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[0], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=2),
        )

        self.branch_2 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[2], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=3, stride=1, padding=1),
            BasicConvBlock(
                in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2),
        )

        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b_1 = self.branch_1(x)
        b_2 = self.branch_2(x)
        b_3 = self.branch_3(x)
        out = torch.cat([b_1, b_2, b_3], dim=1)
        return out


class InceptionBlockE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockE, self).__init__()
        """
        in_channels: int, input_channels
        out_channels: list, [out_channels1reduce,out_channels1,out_channels2reduce, out_channels2]
        """
        self.branch_1 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[0], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=2),
        )

        self.branch_2 = nn.Sequential(
            BasicConvBlock(
                in_channels=in_channels, out_channels=out_channels[2], kernel_size=1),
            BasicConvBlock(
                in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=[1, 7], padding=[0, 3]),
            BasicConvBlock(
                in_channels=out_channels[2], out_channels=out_channels[2], kernel_size=[7, 1], padding=[3, 0]),
            BasicConvBlock(
                in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=3, stride=2),
        )

        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b_1 = self.branch_1(x)
        b_2 = self.branch_2(x)
        b_3 = self.branch_3(x)
        out = torch.cat([b_1, b_2, b_3], dim=1)
        return out


class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv_1 = BasicConvBlock(
            in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv_2 = nn.Conv2d(
            in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear = nn.Linear(
            in_features=768, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv_1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv_2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear(self.auxiliary_dropout(x))
        return out
