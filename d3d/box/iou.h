#pragma once

#include <torch/extension.h>
#include "d3d/box/common.h"

torch::Tensor iou2d_forward(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
void iou2d_backward(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    torch::Tensor grad_boxes1, torch::Tensor grad_boxes2
);
torch::Tensor iou2d_forward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
void iou2d_backward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    torch::Tensor grad_boxes1, torch::Tensor grad_boxes2
);
torch::Tensor iou2dr_forward(
    const torch::Tensor boxes1, const torch::Tensor boxes2, torch::Tensor nx, torch::Tensor xflags
);