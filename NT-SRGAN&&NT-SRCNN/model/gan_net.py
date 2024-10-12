import torch
import torch.nn as nn
from math import sqrt
from .block import *

class Gen_Net(nn.Module):
    def __init__(self):
        super(Gen_Net, self).__init__()
        self.msrb_layer = self.make_layer(MSRB, 3)
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.input_zyx = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True))
        self.relu = nn.ReLU(inplace=True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        z_axis = x[:, 0:1, :, :]
        y_axis = x[:, 1:2, :, :]
        x_axis = x[:, 2:3, :, :]

        out_z_axis = self.input(z_axis)
        out_y_axis = self.input(y_axis)
        out_x_axis = self.input(x_axis)

        out_z_axis = self.msrb_layer(out_z_axis)
        out_y_axis = self.msrb_layer(out_y_axis)
        out_x_axis = self.msrb_layer(out_x_axis)

        out_z_axis = self.output(out_z_axis)
        out_y_axis = self.output(out_y_axis)
        out_x_axis = self.output(out_x_axis)


        out_zyx = torch.cat((out_z_axis, out_y_axis, out_x_axis, z_axis), dim=1)   # 4 channel

        # residual = x
        out_zyx = self.input_zyx(out_zyx)
        out_zyx = self.msrb_layer(out_zyx)
        out_zyx = self.output(out_zyx)

        # out = torch.add(out_zyx, z_axis)

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
        # input_1 = x
        # output_3_1 = self.conv_3_1(input_1)
        # output_5_1 = self.conv_5_1(input_1)
        # input_2 = torch.cat([output_3_1, output_5_1], 1)
        # output_3_2 = self.conv_3_2(input_2)
        # output_5_2 = self.conv_5_2(input_2)
        # input_3 = torch.cat([output_3_2, output_5_2], 1)
        # output = self.confusion(input_3)
        # output += x
        # return self.relu(output)
        input = x
        input = torch.cat([self.conv_3_1(input), self.conv_5_1(input)], 1)
        input = torch.cat([self.conv_3_2(input), self.conv_5_2(input)], 1)
        output = self.confusion(input)
        output += x
        return self.relu(output)

class Dis_Net(nn.Module):
    def __init__(self, LRB_layer_num=3, scale_factor=10, is_init=True):
        super(Dis_Net, self).__init__()
        self.dis_input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        self.residual_layer = self.make_layer(Leaky_Res_Block, LRB_layer_num)
        self.scale_factor = scale_factor
        if self.scale_factor==10:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(1, inplace=True),
                nn.MaxPool2d(5, stride=2),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(1, inplace=True),
                nn.MaxPool2d(3, stride=2),
                nn.Flatten(),
                nn.Linear(1024, 1),
            )
        elif self.scale_factor==5:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(1, inplace=True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(1, inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Flatten(),
                nn.Linear(1600, 1),
            )
        elif self.scale_factor==2:
            self.output = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(1, inplace=True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(1, inplace=True),
                nn.Flatten(),
                nn.Linear(1024, 1),
            )
            
        if is_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    # nn.init.normal_(m.weight, mean=0, std=0.5)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dis_input(x)
        x = self.residual_layer(x)
        x =  self.output(x)
        return x


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).cuda()
    # if torch.cuda.is_available():
	#     alpha = alpha.cuda()
		
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).cuda()
    # if torch.cuda.is_available():
    #     fake = fake.cuda()
    
    gradients = torch.autograd.grad(outputs=d_interpolates,
                                inputs=interpolates,
                                grad_outputs=fake,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True,
                                )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2,dim=1) - 1)**2).mean()
    return gradient_penalty
	
