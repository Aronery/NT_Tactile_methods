import torch.nn as nn


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class Res_Block(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, strides=1):
        super(Res_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        return self.relu(out+x)

class Res_BlockV2(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, strides=1):
        super(Res_BlockV2, self).__init__()
        # k 9 -> 7 -> 5 -> 3
        # p 4    3    2    1
        self.block_1 = nn.Sequential(
            # kernel = 9 padding = 4
            nn.Conv2d(in_channel, out_channel, kernel_size=9, stride=strides, padding=4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # kernel = 7 padding = 3
            nn.Conv2d(out_channel, out_channel, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channel),
            )
        self.block_2 = nn.Sequential(
            # kernel = 5 padding = 2
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=strides, padding=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # kernel = 3 padding = 1
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.block_1(x) + x)
        x = self.relu(self.block_2(x) + x)
        return x

class Leaky_Res_Block(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, strides=1):
        super(Leaky_Res_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(1, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.block(x)
        return self.relu(out+x)