import torch
from torch.nn import Module

from .forward_warp import forward_warp

class forward_warp_rescaled(Module):
    """fowrard warp where input image and warped image are in same value range"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.forward_warp = forward_warp(interpolation_mode="Bilinear")
        self.eps = eps

    def forward(self, im0, flow):

        # get mask
        ones = torch.ones_like(im0)
        mask = self.forward_warp(ones, flow)
        # prevent division by 0
        mask.clamp_(min=self.eps)

        # warp image
        warped = self.forward_warp(im0, flow)

        # rescale image
        rescaled = warped / mask

        return rescaled
