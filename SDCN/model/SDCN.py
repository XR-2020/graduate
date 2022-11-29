import torch.nn
import torchvision
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

    def forward(self, x,ssw):
        image = x  # (1,3,512,512)
        # fE 特征提取器
        feature = self.fE(x)  # VGG16 提取特征 (1,512,32,32)

        # fD 检测分支
        dm,dr=self.fD(feature,ssw)

        # fE分割分支
        s = self.fS(feature)  # 获取分割图   (1,21,512,512)
        # mask = s * image  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # c = self.fC(mask)

        # 协作模块
        max_pro = ssw[:, 1, :]
        ssw = torch.squeeze(ssw, dim=0)
        # 计算所有proposal与得分最高的proposal的iou，torchvision.ops.box_iou数据格式（（N，4），(N，4)）
        iou = torchvision.ops.box_iou(max_pro, ssw)
        print(iou)
        # 转置iou结果
        iou = iou.transpose(0, 1)
        # 交集>0.5会被认为高度重叠，置为1，其余为0,
        iou[iou > 0.5] = 1
        iou[iou < 0.5] = 0
        # 将结果复制20（类别数）份，这样得分最高的proposal以及和它有着高度重叠的proposal也将设为1
        iou = iou.repeat(1, 20)
        # 改变原先的proposal的class
        new_scores = torch.mul(dr, iou)
        print(new_scores)

        # dseg = torch.randn(10,21)
        # d_m = dm * dseg


        return 0

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
    # img = Image.open("000001.jpg")
    # img=Image.new('RGB', (512, 512), (0, 0, 0))
    # transf = transforms.ToTensor()
    # img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
    # img_tensor=torch.unsqueeze(img_tensor,0)
    img_tensor= torch.randn(1, 3, 224, 224)
    ssw_spp = torch.zeros(1, 10, 4)
    for i in range(1):
        for j in range(10):
            ssw_spp[i, j, 0] = 0
            ssw_spp[i, j, 1] = 0
            ssw_spp[i, j, 2] = j + 4
            ssw_spp[i, j, 3] = j + 4
    net = SDCN()
    print(net(img_tensor,ssw_spp))
