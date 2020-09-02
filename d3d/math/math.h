#pragma once

#include "d3d/box/common.h"
#include <torch/extension.h>

torch::Tensor i0e(const torch::Tensor& self);
torch::Tensor i1e(const torch::Tensor& self);

#ifdef BUILD_WITH_CUDA
torch::Tensor i0e_cuda(const torch::Tensor& self);
torch::Tensor i1e_cuda(const torch::Tensor& self);
#endif
