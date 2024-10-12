import torch
import torch.nn as nn
from math import sqrt
from .block import *
from .convLSTM import *


class GenSeqs_Net(nn.Module):
    def __init__(self):
        super(GenSeqs_Net, self).__init__()
        self.msrb_layer = self.make_layer(MSRB, 3)
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.reshape_layer = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.input_zyx = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        # self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        #          bs,axis,seqs,h,w
        out_z_axis = x[:, 0:1, :, :]
        out_y_axis = x[:, 1:2, :, :]
        out_x_axis = x[:, 2:3, :, :]


        # out_z_axis = out_z_axis.view(out_z_axis.shape[0], out_z_axis.shape[1]*out_z_axis.shape[2],out_z_axis.shape[3],out_z_axis.shape[4])
        # out_y_axis = out_y_axis.view(out_y_axis.shape[0], out_y_axis.shape[1]*out_y_axis.shape[2],out_y_axis.shape[3],out_y_axis.shape[4])
        # out_x_axis = out_x_axis.view(out_x_axis.shape[0], out_x_axis.shape[1]*out_x_axis.shape[2],out_x_axis.shape[3],out_x_axis.shape[4])

        # print('!!!', out_x_axis.shape)
        out_z_axis = self.input(out_z_axis)
        out_y_axis = self.input(out_y_axis)
        out_x_axis = self.input(out_x_axis)



        # out_z_axis = self.reshape_layer(out_z_axis)
        # out_y_axis = self.reshape_layer(out_y_axis)
        # out_x_axis = self.reshape_layer(out_x_axis)

        out_z_axis = self.msrb_layer(out_z_axis)
        out_y_axis = self.msrb_layer(out_y_axis)
        out_x_axis = self.msrb_layer(out_x_axis)

        out_z_axis = self.output(out_z_axis)
        out_y_axis = self.output(out_y_axis)
        out_x_axis = self.output(out_x_axis)

        # print(out_x_axis.shape)

        out_zyx = torch.cat((out_z_axis, out_y_axis, out_x_axis), dim=1)   
        # out_zyx = torch.cat((out_z_axis, out_y_axis, out_x_axis,x[:, 0:1, -1, :, :]), dim=1)   

        # # residual = x
        out_zyx = self.input_zyx(out_zyx)
        out_zyx = self.msrb_layer(out_zyx)
        out_zyx = self.output(out_zyx)

        # out = torch.add(out_zyx, z_axis)

        return out_zyx

class GenSeqs_Net_3D(nn.Module):
    def __init__(self):
        super(GenSeqs_Net_3D, self).__init__()
        self.msrb_layer = self.make_layer(MSRB, 3)     
        self.inputSeqs = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),

            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )
        self.reshape_layer = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.input_zyx = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.upSample = nn.Upsample(
            # size=40,
            scale_factor=10,
            mode='bilinear',
            align_corners=False 
        )

        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        #               bs,axis,seqs,h,w
        out_x_axis = self.upSample(x[:, 0, :, :, :])
        out_y_axis = self.upSample(x[:, 1, :, :, :])
        out_z_axis = self.upSample(x[:, 2, :, :, :])

        out_z_axis = out_z_axis.view(out_z_axis.shape[0], 1, out_z_axis.shape[1],
                                    out_z_axis.shape[2], out_z_axis.shape[3])
        out_y_axis = out_y_axis.view(out_z_axis.shape)                  
        out_x_axis = out_x_axis.view(out_z_axis.shape)                          

        out_z_axis = self.inputSeqs(out_z_axis)
        out_y_axis = self.inputSeqs(out_y_axis)
        out_x_axis = self.inputSeqs(out_x_axis)

        out_z_axis = out_z_axis.view(out_z_axis.shape[0],out_z_axis.shape[1]*out_z_axis.shape[2], 
                                        out_z_axis.shape[3], out_z_axis.shape[4])
        out_y_axis = out_y_axis.view(out_z_axis.shape)                  
        out_x_axis = out_x_axis.view(out_z_axis.shape)  
        
        out_z_axis = self.reshape_layer(out_z_axis)
        out_y_axis = self.reshape_layer(out_y_axis)
        out_x_axis = self.reshape_layer(out_x_axis)

        out_z_axis = self.output(out_z_axis)
        out_y_axis = self.output(out_y_axis)
        out_x_axis = self.output(out_x_axis)


        out_zyx = torch.cat((out_z_axis, out_y_axis, out_x_axis), dim=1)   

        out_zyx = self.input_zyx(out_zyx)
        out_zyx = self.msrb_layer(out_zyx)
        out_zyx = self.output(out_zyx)
        return out_zyx

class GenSeqs_Net_LSTM(nn.Module):
    def __init__(self):
        super(GenSeqs_Net_LSTM, self).__init__()
        self.msrb_layer = self.make_layer(MSRB, 3)     
        self.convlstm_layer = CLSTM(shape=(40,40), input_chans=1, 
                                    filter_size=5, num_features=8, num_layers=3)
        
        
        self.reshape_layer = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.input_zyx = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.upSample = nn.Upsample(
            # size=40,
            scale_factor=10,
            mode='bilinear',
            align_corners=False 
        )

        # self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        #               bs,axis,seqs,h,w
        out_x_axis = self.upSample(x[:, 0, :, :, :])
        out_y_axis = self.upSample(x[:, 1, :, :, :])
        out_z_axis = self.upSample(x[:, 2, :, :, :])

        out_z_axis = out_z_axis.view(out_z_axis.shape[0], out_z_axis.shape[1], 1,
                                    out_z_axis.shape[2], out_z_axis.shape[3])
        out_y_axis = out_y_axis.view(out_z_axis.shape)                  
        out_x_axis = out_x_axis.view(out_z_axis.shape)                  
        

        _, out_z_axis = self.convlstm_layer(out_z_axis, self.convlstm_layer.init_hidden(x.shape[0]))
        _, out_y_axis = self.convlstm_layer(out_y_axis, self.convlstm_layer.init_hidden(x.shape[0]))
        _, out_x_axis = self.convlstm_layer(out_x_axis, self.convlstm_layer.init_hidden(x.shape[0]))

        # print(out_z_axis.shape)

        out_z_axis = out_z_axis.reshape(out_z_axis.shape[0],out_z_axis.shape[1]*out_z_axis.shape[2], 
                                        out_z_axis.shape[3], out_z_axis.shape[4])
        out_y_axis = out_y_axis.reshape(out_z_axis.shape)                  
        out_x_axis = out_x_axis.reshape(out_z_axis.shape)  
        
        out_z_axis = self.reshape_layer(out_z_axis)
        out_y_axis = self.reshape_layer(out_y_axis)
        out_x_axis = self.reshape_layer(out_x_axis)


        out_z_axis = self.output(out_z_axis)
        out_y_axis = self.output(out_y_axis)
        out_x_axis = self.output(out_x_axis)

        # print(out_x_axis.shape)

        out_zyx = torch.cat((out_z_axis, out_y_axis, out_x_axis), dim=1)   

        # # residual = x
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

class Dis_Net(nn.Module):
    def __init__(self):
        super(Dis_Net, self).__init__()
        self.dis_input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        self.residual_layer = self.make_layer(Leaky_Res_Block, 3)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(1, inplace=True),
            nn.MaxPool2d(5, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(1, inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 1),
            # nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.dis_input(x)
        x = self.residual_layer(x)
        return self.output(x)


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
	
