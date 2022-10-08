import torch
from math import floor
from spp_layer import spatial_pyramid_pool

BATCH_SIZE=1
R=10
ssw_spp = torch.zeros(BATCH_SIZE, R, 4)
for i in range(BATCH_SIZE):
    for j in range(R):
       ssw_spp[i, j, 0] = 0
       ssw_spp[i, j, 1] = 0
       ssw_spp[i, j, 2] = 4
       ssw_spp[i, j, 3] = 4


def through_spp_new(x,ssw):  # x.shape = [BATCH_SIZE, 512, 14, 14] ssw_get.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
    for i in range(BATCH_SIZE):
        for j in range(ssw.size(1)):
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