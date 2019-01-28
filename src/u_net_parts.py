# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, padding=0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(3, 3),
                padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=(3, 3),
                padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            ),
            DoubleConv(in_ch, out_ch, padding)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, padding=0):
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_ch // 2,
                out_channels=in_ch // 2,
                stride=2)

        self.conv = DoubleConv(in_ch, out_ch, padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = x1 + x2
        #x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
