import torch
import torch.nn as nn
from math import sqrt
from .block import *


class SRCNN_MSRN(nn.Module):
    def __init__(self, feature_layers_num, input_channel=3, scale_factor=10,is_init=True):
        super(SRCNN_MSRN, self).__init__()
        self.msrb_layer = self.make_layer(MSRB, feature_layers_num)
        self.input_zyx = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.upSample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False 
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True))
        self.relu = nn.ReLU(inplace=True)

        if is_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out_zyx = self.upSample(x)
        out_zyx = self.input_zyx(out_zyx)
        out_zyx = self.msrb_layer(out_zyx)
        out_zyx = self.output(out_zyx)

        return out_zyx

class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
        )

        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size_2, padding=2),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
        )

        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size_1, padding=1),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(True),
        )

        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size_2, padding=2),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(True),
        )

        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(input_1)
        output_5_1 = self.conv_5_1(input_1)
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return self.relu(output)
# """

if __name__ == '__main__':
    model=MSRB()
    print(model)

