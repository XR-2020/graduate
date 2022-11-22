import torch
from torch import nn
from torch.nn import functional as F


class Seg(nn.Module):#init seg

    def __init__(self):
        super(Seg, self).__init__()

        # 解码部分
        self.c1 = Conv(512, 256)
        self.up1 = UpSample(256)

        self.c2 = Conv(256, 128)
        self.up2 = UpSample(128)

        self.c3 = Conv(128, 64)
        self.up3 = UpSample(64)

        self.c4 = Conv(64, 32)
        self.up4 = UpSample(32)

        self.c5 = Conv(32, 21)
        # 归一化
        self.Th = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.c1(x)
        x = self.up1(x)
        x = self.c2(x)
        x = self.up2(x)
        x = self.c3(x)
        x = self.up3(x)
        x = self.c4(x)
        x = self.up4(x)

        x = self.c5(x)

        return self.Th(x)

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):

    def __init__(self, C):
        super(UpSample, self).__init__()
        # 特征图大小扩大2倍，通道数不变
        self.Up = nn.Conv2d(C, C, 1, 1)

    def forward(self, x):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)

        return x
