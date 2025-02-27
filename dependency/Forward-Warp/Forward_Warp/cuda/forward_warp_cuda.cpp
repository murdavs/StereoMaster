#include <torch/torch.h>
#include <vector>

#include "forward_warp.h"
using at::native::detail::GridSamplerInterpolation;

// --------------------------------------------------------------------------
// Existing forward/backward/max_motion declarations
// --------------------------------------------------------------------------
at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const GridSamplerInterpolation interpolation_mode);

std::vector<at::Tensor> forward_warp_cuda_backward(
    const at::Tensor grad_output,
    const at::Tensor im0,
    const at::Tensor flow,
    const GridSamplerInterpolation interpolation_mode);

at::Tensor forward_warp_max_motion_cuda_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const at::Tensor im1,
    const at::Tensor d_buffer,
    const at::Tensor wght_buffer);

// --------------------------------------------------------------------------
// The existing wrappers (DO NOT remove anything)
// --------------------------------------------------------------------------
at::Tensor forward_warp_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const int interpolation_mode){
  return forward_warp_cuda_forward(im0, flow, (GridSamplerInterpolation)interpolation_mode);
}

std::vector<at::Tensor> forward_warp_backward(
    const at::Tensor grad_output,
    const at::Tensor im0,
    const at::Tensor flow,
    const int interpolation_mode){
  return forward_warp_cuda_backward(grad_output, im0, flow, (GridSamplerInterpolation)interpolation_mode);
}

at::Tensor forward_warp_max_motion_forward(
    const at::Tensor im0,
    const at::Tensor flow,
    const at::Tensor im1,
    const at::Tensor d_buffer,
    const at::Tensor wght_buffer){
  return forward_warp_max_motion_cuda_forward(im0, flow, im1, d_buffer, wght_buffer);
}

// --------------------------------------------------------------------------
// NEW CODE: forward_warp_forward_mask
// --------------------------------------------------------------------------
/*
   This new function calls a CUDA kernel (you must implement it in 
   forward_warp_cuda_kernel.cu or similar) that accumulates color in 'im1' 
   and also accumulates weight in 'mask1'. The function returns a vector
   with two tensors: [im1, mask1].
*/
std::vector<at::Tensor> forward_warp_forward_mask(
    const at::Tensor im0,
    const at::Tensor flow,
    const int interpolation_mode);

// --------------------------------------------------------------------------
// PYBIND module: we add a "forward_mask" entry
// --------------------------------------------------------------------------
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME, 
    m){
  // Existing bindings
  m.def("forward", &forward_warp_forward, "forward warp forward (CUDA)");
  m.def("backward", &forward_warp_backward, "forward warp backward (CUDA)");
  m.def("forward_max_motion", &forward_warp_max_motion_forward, "forward warp max motion forward (CUDA)");

  // New binding
  m.def("forward_mask", &forward_warp_forward_mask, "forward warp + mask (CUDA)");
}

// --------------------------------------------------------------------------
// Implementation of forward_warp_forward_mask (the new function)
// --------------------------------------------------------------------------

// Declare a function in your .cu to actually run the kernel that 
// accumulates color and mask:
at::Tensor forward_warp_cuda_forward_mask(
    const at::Tensor im0,
    const at::Tensor flow,
    at::Tensor im1,
    at::Tensor mask1,
    GridSamplerInterpolation interpolation_mode
);

std::vector<at::Tensor> forward_warp_forward_mask(
    const at::Tensor im0,
    const at::Tensor flow,
    const int interpolation_mode)
{
  // We create im1 and mask1
  auto im1 = torch::zeros_like(im0);
  int B = im0.size(0);
  int H = im0.size(2);
  int W = im0.size(3);

  // mask1 => shape [B, 1, H, W], same device & type as im0
  auto mask1 = torch::zeros({B,1,H,W}, im0.options());

  // Flatten mask1 to [B*H*W] if your kernel expects a 1D buffer
  // (depending on your implementation in the .cu)
  at::Tensor mask1_flat = mask1.view({B*H*W});

  // Now call the function that launches the kernel
  forward_warp_cuda_forward_mask(
    im0,
    flow,
    im1,
    mask1_flat,
    (GridSamplerInterpolation)interpolation_mode
  );

  // Return [im1, mask1]
  // so Python can do: im1, mask1 = forward_mask(im0, flow, ...)
  return { im1, mask1 };
}
