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

    def forward(self, x, ssw, label,BCE_function,CE_function):
        image = x  # (1,3,512,512)
        # fE 特征提取器
        feature = self.fE(x)  # VGG16 提取特征 (1,512,32,32)

        # fD 检测分支
        dm, dr,score = self.fD(feature, ssw)
        # 计算检测多分类损失LDmil
        LDmil=BCE_function(score,label)


        # fE分割分支
        s = self.fS(feature)  # 获取分割图   (1,21,512,512)
        # 分离RGB通道，每个通道依次 * 每一类的分割图，循环21次，依次处理每个类别,得出I在每类mask处理后的每类分类得分cls_score
        R = x[:, 0, :, :]
        G = x[:, 1, :, :]
        B = x[:, 2, :, :]
        for k in range(21):
            # I ∗ sk
            channel_R = R * s[:, k, :, :]
            channel_G = G * s[:, k, :, :]
            channel_B = B * s[:, k, :, :]
            channel_RGB = torch.stack([channel_R, channel_G, channel_B]).transpose(0, 1)
            label_sk = torch.zeros(label.shape)
            label_sk[:,k] = 1
            # I ∗ ( 1-sk )
            channel_R = R * (1 - s[:, k, :, :])
            channel_G = G * (1 - s[:, k, :, :])
            channel_B = B * (1 - s[:, k, :, :])
            channel_1_RGB = torch.stack([channel_R, channel_G, channel_B]).transpose(0, 1)
            label_1_sk = label
            label_1_sk[:,k] = 0
            # 与标签进行loss计算  LBCE(f C(I ∗ sk), ˜y) + LBCE(f C(I ∗ (1 − sk)), ˆy)
            L_adv = BCE_function(self.fC(channel_RGB), label_sk) +BCE_function(self.fC(channel_1_RGB), label_1_sk)

        # 协作模块
        """"# Detection 指导 Segmentation
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
        """
        # Detection 指导 Segmentation
        # 初始化S_det,全黑
        S_det=torch.zeros(10,x.shape[2],x.shape[3])


        # Segmentation 指导 Detection  
        dseg = torch.rand(10,21)
        d_m = dm * dseg
        proposals = torch.squeeze(ssw, dim=0)
        # 随意初始化伪标签
        Y_r = torch.zeros(d_m.shape[1],1)
        #  每次都计算所有框与得分最高的框的overlap
        for i in range(d_m.shape[2]):
            # 找出第i类得分最高的proposal的下标
            index = torch.max(d_m[:, :, i], dim=1).indices.item()
            # 计算其他框与最高分框的iou
            iou = torchvision.ops.box_iou(ssw[:,index,:], proposals)
            # 行列转换，形成第i类的伪标签
            iou = iou.transpose(0, 1)
            iou[iou > 0.5] = 1
            iou[iou < 0.5] = 0
            # 拼接所有类的伪标签Y_r
            if i == 0:
                Y_r = iou
            else:
                Y_r = torch.cat((Y_r,iou),1)
        # 伪标签升维与D、dr保持一致，用于loss计算
        Y_r=Y_r.unsqueeze(0)
        LDref=CE_function(dr,Y_r)




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
    BCE_function = nn.BCELoss(weight=None, reduction='mean')
    CE_function = nn.CrossEntropyLoss(weight=None, reduction='mean')
    img_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]],dtype=torch.float32)
    ssw_spp = torch.zeros(1, 10, 4)
    for i in range(1):
        for j in range(10):
            ssw_spp[i, j, 0] = 0
            ssw_spp[i, j, 1] = 0
            ssw_spp[i, j, 2] = j + 4
            ssw_spp[i, j, 3] = j + 4
    net = SDCN()
    print(net(img_tensor, ssw_spp, label,BCE_function,CE_function))
