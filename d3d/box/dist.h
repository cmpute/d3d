#pragma once

#include <tuple>
#include <torch/extension.h>
#include "d3d/box/common.h"

std::tuple<torch::Tensor /*distance*/, torch::Tensor /*iedge*/> pdist2dr_forward(
    const torch::Tensor points, const torch::Tensor boxes
);
std::tuple<torch::Tensor /*grad_boxes*/, torch::Tensor /*grad_points*/> pdist2dr_backward(
    const torch::Tensor points, const torch::Tensor boxes, const torch::Tensor grad, const torch::Tensor iedge
);

#ifdef BUILD_WITH_CUDA

std::tuple<torch::Tensor /*distance*/, torch::Tensor /*iedge*/> pdist2dr_forward_cuda(
    const torch::Tensor points, const torch::Tensor boxes
);
std::tuple<torch::Tensor /*grad_boxes*/, torch::Tensor /*grad_points*/> pdist2dr_backward_cuda(
    const torch::Tensor points, const torch::Tensor boxes, const torch::Tensor grad, const torch::Tensor iedge
);

#endif
