from __future__ import print_function
import torch
from torch import nn
import numpy as np

_baseline = 0.54
def disp2depth(calib, disp):
    return calib * _baseline / np.clip(disp, 1, None)

def depth2disp(calib, depth):
    return calib * _baseline / np.clip(depth, 1, None)

def disp2depth_torch(calib, disp):
    return calib * _baseline / torch.clamp(disp, min=1)

def depth2disp_torch(calib, depth):
    return calib * _baseline / torch.clamp(depth, min=1)

def convert_cost_volume(cost, calib, depths):
    costD = torch.FloatTensor(*cost.size()).zero_().cuda()
    for b in range(cost.size()[0]):
        for i, d in enumerate(depths[b]):
            disp = depth2disp(calib[b].cpu().numpy(), d)
            lower = int(disp)
            upper = lower + 1;
            converted = (cost[b,:,lower,:,:] * (upper-disp) + cost[b,:,upper,:,:] * (disp-lower))
            costD[b,:,i,:,:] = converted
    return costD

class depthregression(nn.Module):
    def __init__(self, depths):
        super(depthregression, self).__init__()
        B, D = depths.shape
        self.depths = torch.Tensor(depths).cuda().view(B, D, 1, 1)
    def forward(self, x):
        B, D, H, W = x.size()
        depth = self.depths.expand(B, D, H, W)
        out = torch.sum(x * depth, 1)
        return out

    
