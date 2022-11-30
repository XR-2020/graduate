import math
from math import floor

import torch
from torch import nn
import torch.nn.functional as F

class Dec(nn.Module):#init Dec

    def __init__(self):
        super(Dec, self).__init__()

        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, 21)
        self.fc8d = nn.Linear(4096, 21)

    def forward(self, x, ssw_get):
        x = self.through_spp_new(x, ssw_get)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x_c = F.relu(self.fc8c(x))
        x_d = F.relu(self.fc8d(x))
        dr = F.softmax(x_c, dim=2)  # 按class进行softmax
        dm = F.softmax(x_d, dim=1)  # 按proposal进行softmax
        dm = dr * dm
        score = torch.sum(dm, dim=1)

        return dm, dr,score

    def through_spp_new(self, x,ssw):  # x.shape = [BATCH_SIZE, 512, 14, 14] ssw_get.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
        for i in range(1):
            for j in range(ssw.size(1)):
                '''
                    ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    1.fmap_piece = torch.unsqueeze(x[i, :, floor(ssw[i, j, 0]) : floor(ssw[i, j, 0] + ssw[i, j, 2]), 
                                      floor(ssw[i, j, 1]) : floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                      是按照ssw中给出的四个数字即左上右下坐标点截取原图，floor:floor即为截取指定行指定列的原图，[a,b,floor:floor,floor:floor]
                '''
                fmap_piece = torch.unsqueeze(x[i, :, floor(ssw[i, j, 0]): floor(ssw[i, j, 0] + ssw[i, j, 2]),
                                             floor(ssw[i, j, 1]): floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                fmap_piece = spatial_pyramid_pool(previous_conv=fmap_piece, num_sample=1,
                                                  previous_conv_size=[fmap_piece.size(2), fmap_piece.size(3)],
                                                  out_pool_size=[2, 2])
                if j == 0:
                    y_piece = fmap_piece
                    # print('fmap_piece.shape', fmap_piece.shape)
                else:

                    y_piece = torch.cat((y_piece, fmap_piece))
            if i == 0:
                y = torch.unsqueeze(y_piece, 0)
                # print('y_piece', y_piece.shape)
            else:
                y = torch.cat((y, torch.unsqueeze(y_piece, 0)))
        return y


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i])
        w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i])
        h_pad = min(math.floor((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2), math.floor(h_wid / 2))
        w_pad = min(math.floor((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2), math.floor(w_wid / 2))
        # print([h_wid,w_wid,h_pad,w_pad])
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            '''
              把特征拉成向量
            '''
            spp = x.view(num_sample, -1)  # 某一个维度传入数字-1，表示自动对维度进行计算并变化
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            '''
                把特征向量拼接起来
            '''
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)  # 按照第一个维度cat
    return spp