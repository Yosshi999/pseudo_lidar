from __future__ import print_function
import torch
from torch import nn
import numpy as np

_baseline = 0.54
def disp2depth(calib, disp):
    return calib * _baseline / np.clip(disp, 1, None)

def depth2disp(calib, depth):
    return calib * _baseline / np.clip(depth, 1, None)

def convert_cost_volume(cost, calib, depths):
    costD = torch.FloatTensor(*cost.size()).zero_()
    for i, d in enumerate(depths):
        disp = depth2disp(calib, d)
        lower = int(disp)
        upper = lower + 1;
        converted = (cost[:,:,lower,:,:] * (upper-disp) + cost[:,:,upper,:,:] * (disp-lower))
        costD[:,:,i,:,:] = converted
    return costD

class depthregression(nn.Module):
    def __init__(self, depths):
        super().__init__()
        self.depths = torch.Tensor(depths).cuda().view(1, len(depths), 1, 1)
    def forward(self, x):
        B, D, H, W = x.size()
        depth = self.depths.repeat(B, 1, H, W)
        out = torch.sum(x * depth, 1)
        return out

    
