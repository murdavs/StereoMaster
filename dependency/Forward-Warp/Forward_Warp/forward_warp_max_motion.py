import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

import forward_warp_cuda
from .python import Forward_Warp_Python


class forward_warp_max_motion_function(Function):

    @staticmethod
    def forward(ctx, im0, flow, eps):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)
        assert(im0.is_contiguous())
        assert(flow.is_contiguous())
        assert(torch.isnan(flow).long().sum() == 0)
        assert(torch.isinf(flow).long().sum() == 0)

        B, C, H, W = im0.shape
        im1_buffer = torch.zeros_like(im0)
        d_buffer = torch.zeros(B, 1, H, W, dtype=torch.int32, device=im0.device)
        wght_buffer = torch.zeros(B, 1, H, W, device=im0.device)

        ctx.save_for_backward(im0, flow)
        if im0.is_cuda:
            im1_buffer = forward_warp_cuda.forward_max_motion(im0, flow, im1_buffer, d_buffer, wght_buffer)
        else:
            raise NotImplementedError

        # wght is added C times to buffer
        wght_buffer /= C

        # rescale image
        im1 = im1_buffer / wght_buffer.clamp(min=eps)

        # disocclusion
        disocclusions = torch.zeros(B, 1, H, W, device=im0.device)
        disocclusions[wght_buffer == 0] = 1

        return im1, disocclusions, im1_buffer, d_buffer, wght_buffer
 
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class forward_warp_max_motion(Module):
    '''
    Adapted from Algorithm 3 in Sanachez et al. 2013 "Computing Inverse Optical Flow".
    Note that this algorithm only warps forward and does not invert result of forward warp.
    Multiply with -1 to get same results as Sanachez et al.
    '''

    def __init__(self, eps=1e-5):
        super(forward_warp_max_motion, self).__init__()
        self.eps = eps

    def forward(self, im0, flow, return_disocclusions=False, debug=False):

        im1, disocclusions, im1_buffer, d_buffer, wght_buffer = forward_warp_max_motion_function.apply(im0, flow, self.eps)

        if debug:
            return im1, disocclusions, im1_buffer, d_buffer, wght_buffer
        elif return_disocclusions:
            return im1, disocclusions
        else:
            return im1
