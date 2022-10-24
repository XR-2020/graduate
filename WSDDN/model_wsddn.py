import torch
import torchvision.models as v_models
import torch.nn as nn
import torch.nn.functional as F
from math import floor

from spp_layer import spatial_pyramid_pool
#from data_pre import myDataSet

BATCH_SIZE = 1
R = 10
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

#最终在进行SPP时只保留10个proposals参与网络的运算，最后生成20类的概率
class WSDDN(nn.Module):
    def __init__(self, vgg_name):
        super(WSDDN, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)

    def forward(self, x, ssw_get): #x.shape = [BATCH_SIZE, 3, h, w]  ssw_get.shape = [BATCH_SIZE, R, 4] out.shape = [BATCH_SIZE, 20]
        x = self.features(x)#搭建出backboneVGG11的前5层（1,512,14,14）
        x = self.through_spp_new(x, ssw_get)#SPP层（1，10,4096）
        #print(x.shape)
        #out = out.view(out.size(0), -1)
        #x = self.through_spp(x)
        #print(x.shape)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x_c = F.relu(self.fc8c(x))
        x_d = F.relu(self.fc8d(x))
        #print(x_c.shape)
        #print(x_d)
        segma_c = F.softmax(x_c, dim = 2)#按类别softmax
        segma_d = F.softmax(x_d, dim = 1)#按proposal进行softmax
        #print(segma_c)
        #print(segma_d)
        #print(segma_c.shape)
        #print(segma_d.shape)
        x = segma_c * segma_d
        x = torch.sum(x, dim = 1)
        #print(x.shape)
        return x, segma_d, segma_c

    def _make_layers(self, cfg):  #init VGG
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def through_spp_new(self, x, ssw):  #x.shape = [BATCH_SIZE, 512, 14, 14] ssw_get.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
        for i in range(BATCH_SIZE):
            for j in range(ssw.size(1)):
                '''
                    1.fmap_piece = torch.unsqueeze(x[i, :, floor(ssw[i, j, 0]) : floor(ssw[i, j, 0] + ssw[i, j, 2]), 
                                      floor(ssw[i, j, 1]) : floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                      是按照ssw中给出的四个数字即左上坐标点以及bpundingbox宽和高截取原图，floor:floor即为截取指定行指定列的原图，[a,b,floor:floor,floor:floor]
                    2.spatial_pyramid_pool[2,2]无论输入的尺寸大小统一通过pool转为大小为2 * 2的大小，然后将其拉成向量
                '''
                fmap_piece = torch.unsqueeze(x[i, :, floor(ssw[i, j, 0]) : floor(ssw[i, j, 0] + ssw[i, j, 2]), 
                                      floor(ssw[i, j, 1]) : floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                #让尺寸大小不定的bounding box通过金字塔池化转为指定尺寸,尺寸分别为1*1,2*2,3*3
                fmap_piece = spatial_pyramid_pool(previous_conv = fmap_piece, num_sample = 1,
                                        previous_conv_size = [fmap_piece.size(2),fmap_piece.size(3)], out_pool_size = [1,2,3])
                if j == 0:
                    y_piece = fmap_piece
                    #print('fmap_piece.shape', fmap_piece.shape)
                else:

                    y_piece = torch.cat((y_piece, fmap_piece))
            if i == 0:
                y = torch.unsqueeze(y_piece, 0)
                #print('y_piece', y_piece.shape)
            else:
                y = torch.cat((y, torch.unsqueeze(y_piece, 0)))
        return y

    def through_spp(self, x):  #spp_layer
        for i in range(BATCH_SIZE):
            y_piece = torch.unsqueeze(spatial_pyramid_pool(previous_conv = x[i,:], num_sample = R, 
                                        previous_conv_size = [x.size(3),x.size(4)], out_pool_size = [2, 2]), 0)
            if i == 0:
                y = y_piece
                #print(y_piece.shape)
            else:
                y = torch.cat((y, y_piece))
                #print(y.shape)
        return y

    def select_fmap(self, fmap, ssw): #choose interested region  fmap.shape = [BATCH_SIZE, 512, 14, 14]  ssw.shape = [BATCH_SIZE, R, 4]
        for i in range(BATCH_SIZE):
            for j in range(ssw.size(1)):
                fmap_piece = torch.unsqueeze(fmap[i, :, floor(ssw[i, j, 0]) : floor(ssw[i, j, 0] + ssw[i, j, 2]), 
                                      floor(ssw[i, j, 1]) : floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                if j == 0:
                    y_piece = fmap_piece
                    #print(y_piece.shape)
                else:
                    y_piece = torch.cat((y_piece, fmap_piece), 0)
                    #print(y_piece.shape)
            if i == 0:
                y = torch.unsqueeze(y_piece, 0)
            else:
                y = torch.cat((y, torch.unsqueeze(y_piece, 0)), 0)
        return y

if __name__ == '__main__':
    net_test = WSDDN('VGG11')
    x_test = torch.randn(BATCH_SIZE, 3, 224, 224)
    # ssw_spp = torch.zeros(BATCH_SIZE, R, 4)
    ssw_spp = torch.zeros(BATCH_SIZE, R, 4)
    for i in range(BATCH_SIZE):
        for j in range(R):
            ssw_spp[i, j, 0] = i+5
            ssw_spp[i, j, 1] = j+6
            ssw_spp[i, j, 2] = i+j+10
            ssw_spp[i, j, 3] = i+j+14
    out_test = net_test(x_test, ssw_spp)
    #print(out_test[0].shape)
    print(out_test[0])#20分类每类的概率
    '''
    ssw_spp = torch.zeros(BATCH_SIZE, R, 4)
    for i in range(BATCH_SIZE):
        for j in range(R):
            ssw_spp[i, j, 0] = 0
            ssw_spp[i, j, 1] = 0
            ssw_spp[i, j, 2] = 4
            ssw_spp[i, j, 3] = 4
    map_test = torch.randn(BATCH_SIZE, 512, 14, 14)
    y_test = select_fmap(map_test, ssw_spp)
    print(y_test.shape)
    '''
    
    '''
    spp_test = torch.randn(BATCH_SIZE, R, 512, 14, 14)
    out_test = through_spp(spp_test)
    print(out_test.shape)
    '''
#pretrained_model_path = 
#net_wsddn = WSDDN('VGG11')
#state_dict = torch.load(pretrained_model_path)
#net_wsddn.load_state_dict({k: v for k, v in state_dict.items() if k in net_wsddn.state_dict()})
