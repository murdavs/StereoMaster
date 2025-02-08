#include <torch/torch.h>
#include <vector>

#include "forward_warp.h"
using at::native::detail::GridSamplerInterpolation;

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

// Because of the incompatible of Pytorch 1.0 && Pytorch 0.4, we have to annotation this.
// #define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward_warp_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const int interpolation_mode){
  // CHECK_INPUT(im0);
  // CHECK_INPUT(flow);
  return forward_warp_cuda_forward(im0, flow, (GridSamplerInterpolation)interpolation_mode);
}

std::vector<at::Tensor> forward_warp_backward(
    const at::Tensor grad_output,
    const at::Tensor im0,
    const at::Tensor flow,
    const int interpolation_mode){
  // CHECK_INPUT(grad_output);
  // CHECK_INPUT(im0);
  // CHECK_INPUT(flow);
  return forward_warp_cuda_backward(grad_output, im0, flow, (GridSamplerInterpolation)interpolation_mode);
}

at::Tensor forward_warp_max_motion_forward(
    const at::Tensor im0,
    const at::Tensor flow,
    const at::Tensor im1,
    const at::Tensor d_buffer,
    const at::Tensor wght_buffer){
  // CHECK_INPUT(im0);
  // CHECK_INPUT(flow);
  // CHECK_INPUT(im1);
  // CHECK_INPUT(d_buffer);
  // CHECK_INPUT(wght_buffer);
  return forward_warp_max_motion_cuda_forward(im0, flow, im1, d_buffer, wght_buffer);
}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME, 
    m){
  m.def("forward", &forward_warp_forward, "forward warp forward (CUDA)");
  m.def("backward", &forward_warp_backward, "forward warp backward (CUDA)");
  m.def("forward_max_motion", &forward_warp_max_motion_forward, "forward warp max motion forward (CUDA)");
}
