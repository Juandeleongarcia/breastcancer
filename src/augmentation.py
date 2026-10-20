"""Una sola geometría sobre los tres canales, sin cambios fotométricos."""
import math

import torch
import torch.nn.functional as F


def synchronized_affine(image, cfg):
    if torch.rand(()).item() >= cfg['AUGMENT_PROBABILITY']:
        return image
    angle = math.radians((2*torch.rand(()).item()-1)*cfg['AUGMENT_DEGREES'])
    shift = (2*torch.rand(2)-1)*cfg['AUGMENT_TRANSLATE']*2
    scale = 1+(2*torch.rand(()).item()-1)*cfg['AUGMENT_SCALE_DELTA']
    theta = image.new_tensor([[math.cos(angle)/scale, -math.sin(angle)/scale, float(shift[0])],
                              [math.sin(angle)/scale, math.cos(angle)/scale, float(shift[1])]])
    grid = F.affine_grid(theta.unsqueeze(0), (1,*image.shape), align_corners=False)
    return F.grid_sample(image.unsqueeze(0),grid,mode='bilinear',padding_mode='reflection',
                         align_corners=False).squeeze(0)
