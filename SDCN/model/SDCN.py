import torch.nn
from PIL import Image
from torch import nn
from Segementation import Seg
from Classification import Cls
from Detection import Dec
import torchvision.transforms as transforms
import cv2 as cv

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
}


class SDCN(nn.Module):
    def __init__(self):
        super(SDCN, self).__init__()
        self.fE = self.make_layers(cfg['VGG16'])
        self.fS = Seg()
        self.fC = Cls(21)
        self.fD = Dec()

    def forward(self, x):
        image = x  # (1,3,512,512)
        feature = self.fE(x)  # VGG16 提取特征 (1,512,32,32)
        s = self.fS(feature)  # 获取分割图   (1,21,512,512)
        s=torch.randn(1,21,512,512)
        cls_s = s/255
        for i in range(1,s.shape[1]):
            mask = cls_s[:, i, :, :] * image
            c = self.fC(mask)
            print(c)



        return s

    def make_layers(self, cfg):  # init VGG
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


if __name__ == '__main__':
    img = Image.open("000001.jpg")
    img=Image.new('RGB', (512, 512), (0, 0, 0))
    transf = transforms.ToTensor()
    img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
    img_tensor=torch.unsqueeze(img_tensor,0)
    net = SDCN()
    print(net(img_tensor).shape)
