#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_mask(img1, img2, mask):
    """PSNR over the pixels where mask == 1 (PSNR-F when mask is the object mask)."""
    mask = mask.view(mask.shape[0], -1)
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1) * mask
    mse = mse.sum(1) / mask.sum(1)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
