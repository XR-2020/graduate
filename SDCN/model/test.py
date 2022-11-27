# import math
#
# import cv2
# import selectivesearch
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
#
# # 第二步：执行搜索工具,展示搜索结果
# image2 = "000005.jpg"
#
# # 用cv2读取图片
# img = cv2.imread(image2)
#
# # # 白底黑字图 改为黑底白字图
# # img = 255 - img
#
# # selectivesearch 调用selectivesearch函数 对图片目标进行搜索
# # img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.7, min_size=20)
# def ssw(img,scale=500,sigma=0.7,min_size=20):
#     img_lbl,regions=selectivesearch.selective_search(img,scale=scale,sigma=sigma,min_size=min_size)
#     candidates =set()
#     for r in regions:
#         # 重复的不要
#         if r['rect'] in candidates:
#             continue
#         # # 太大的不要
#         if r['size'] > 2000 or r['size']<5:
#             continue
#         #x, y, w, h = r['rect']
#         # 太不方的不要
#         #if w  > 2*h or h > 2* w :
#         #    continue
#         candidates.add(r['rect'])
#     return list(candidates)
#
# regions = ssw(img, scale=500, sigma=0.7, min_size=20)
# mapping=[]
# for ele in regions:
#     print("__________________________________________________-")
#     # ceil向上取整，floor向下取整
#     mapping.append((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
#                     math.ceil((ele[0] + ele[2]) / 16) - 1 - (math.floor(ele[0] / 16) + 1),
#                     math.ceil((ele[1] + ele[3]) / 16) - 1 - (math.floor(ele[1] / 16) + 1)))
#     print(ele)
#     print("************")
#     print((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
#                     math.ceil((ele[0] + ele[2]) / 16) - 1 - (math.floor(ele[0] / 16) + 1),
#                     math.ceil((ele[1] + ele[3]) / 16) - 1 - (math.floor(ele[1] / 16) + 1)))
#     print("************")
#     print((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
#                                     math.ceil(ele[2] / 16),
#                                     math.ceil(ele[3] / 16)))
#     # mapping.append((math.floor(ele[0] / 16) + 1, math.floor(ele[1] / 16) + 1,
#     #                                 math.ceil(ele[2] / 16),
#     #                                 math.ceil(ele[3] / 16)))
# mapping = list(set(mapping))
#
# # # 接下来我们把窗口和图像打印出来，对它有个直观认识
# # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
# # ax.imshow(img)
# # for reg in mapping:
# #     x, y, w, h = reg
# #     rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
# #     # if reg['size']<500 or reg['size']>1000:
# #     #     continue
# #     ax.add_patch(rect)
# #
# # plt.show()
#
'''
根据WSDDN得出的proposal、class计算各proposal和max score proposal的iou
进一步修改用于生成heatmap的最终scores
'''
import math

import torch
import torchvision

if __name__ == '__main__':
    scores=torch.tensor([[[0.0504, 0.0550, 0.0476, 0.0476, 0.0556, 0.0476, 0.0476, 0.0476,
          0.0476, 0.0544, 0.0518, 0.0535, 0.0476, 0.0476, 0.0522, 0.0476,
          0.0476, 0.0514, 0.0500, 0.0496],
         [0.0480, 0.0584, 0.0482, 0.0471, 0.0577, 0.0471, 0.0471, 0.0471,
          0.0471, 0.0542, 0.0525, 0.0549, 0.0471, 0.0471, 0.0527, 0.0471,
          0.0471, 0.0522, 0.0502, 0.0471],
         [0.0483, 0.0584, 0.0462, 0.0461, 0.0553, 0.0461, 0.0461, 0.0461,
          0.0461, 0.0585, 0.0553, 0.0504, 0.0461, 0.0461, 0.0609, 0.0467,
          0.0461, 0.0527, 0.0525, 0.0461],
         [0.0492, 0.0608, 0.0457, 0.0457, 0.0523, 0.0457, 0.0457, 0.0457,
          0.0457, 0.0590, 0.0551, 0.0480, 0.0457, 0.0457, 0.0610, 0.0457,
          0.0457, 0.0542, 0.0573, 0.0457],
         [0.0493, 0.0567, 0.0448, 0.0448, 0.0555, 0.0457, 0.0448, 0.0448,
          0.0448, 0.0652, 0.0573, 0.0474, 0.0448, 0.0448, 0.0616, 0.0474,
          0.0448, 0.0543, 0.0565, 0.0448],
         [0.0492, 0.0550, 0.0449, 0.0449, 0.0530, 0.0469, 0.0449, 0.0449,
          0.0449, 0.0631, 0.0585, 0.0487, 0.0449, 0.0449, 0.0605, 0.0500,
          0.0449, 0.0558, 0.0552, 0.0449],
         [0.0476, 0.0579, 0.0449, 0.0449, 0.0517, 0.0453, 0.0449, 0.0449,
          0.0449, 0.0626, 0.0556, 0.0527, 0.0449, 0.0449, 0.0607, 0.0495,
          0.0449, 0.0559, 0.0560, 0.0449],
         [0.0482, 0.0581, 0.0450, 0.0450, 0.0529, 0.0450, 0.0450, 0.0450,
          0.0450, 0.0619, 0.0561, 0.0523, 0.0450, 0.0450, 0.0609, 0.0488,
          0.0450, 0.0551, 0.0554, 0.0450],
         [0.0474, 0.0603, 0.0449, 0.0449, 0.0548, 0.0449, 0.0449, 0.0449,
          0.0449, 0.0658, 0.0554, 0.0526, 0.0449, 0.0449, 0.0588, 0.0449,
          0.0449, 0.0537, 0.0569, 0.0449],
         [0.0469, 0.0593, 0.0446, 0.0446, 0.0545, 0.0446, 0.0446, 0.0446,
          0.0446, 0.0658, 0.0561, 0.0555, 0.0446, 0.0446, 0.0640, 0.0446,
          0.0446, 0.0532, 0.0535, 0.0446]]])

    #模拟proposal
    ssw_spp = torch.zeros(1, 10, 4)
    for i in range(1):
        for j in range(10):
            ssw_spp[i, j, 0] = 0
            ssw_spp[i, j, 1] = 0
            ssw_spp[i, j, 2] = j + 4
            ssw_spp[i, j, 3] = j + 4
    max_pro = ssw_spp[:,1,:]
    ssw_spp = torch.squeeze(ssw_spp, dim=0)

    #计算所有proposal与得分最高的proposal的iou，torchvision.ops.box_iou数据格式（（N，4），(N，4)）
    iou=torchvision.ops.box_iou(max_pro,ssw_spp)

    #转置iou结果
    iou=iou.transpose(0,1)

    #交集>0.5会被认为高度重叠，置为1，其余为0,
    iou[iou > 0.5] = 1
    iou[iou < 0.5] = 0

    # 将结果复制20（类别数）份，这样得分最高的proposal以及和它有着高度重叠的proposal也将设为1
    iou = iou.repeat(1, 20)

    #改变原先的proposal的class
    new_scores=torch.mul(scores,iou)
    print(new_scores)
    # ssw_spp=torch.squeeze(ssw_spp, dim=0)
    #
    # for i in range(ssw_spp.shape[0]):
    #     rect1 = tuple(ssw_spp[i, :].tolist())
    #     for j in range(ssw_spp.shape[0]):
    #         rect2 = tuple(ssw_spp[j, :].tolist())
    #         print("________________________")
    #         print(rect1)
    #         print(rect2)
    #         print(calc_area(rect1,rect2))

