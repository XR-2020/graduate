import math
import torch.nn as nn
import torch


#(1,512,4,4),1,(list[4,4]),(list[2,2])
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
        # pytorch要求 padding 不超过 kernel 的一半，所以要有min
        h_pad = min(math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2),math.floor(h_wid/2))
        w_pad = min(math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2),math.floor(w_wid/2))
        #print([h_wid,w_wid,h_pad,w_pad])
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            '''
              把特征拉成向量
            '''
            spp = x.view(num_sample,-1)#某一个维度传入数字-1，表示自动对维度进行计算并变化
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            '''
                把特征向量拼接起来
            '''
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)#按照第一个维度cat
    return spp
