#pragma once

#include <torch/extension.h>

torch::Tensor rbox_2d_iou(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
torch::Tensor rbox_2d_iou_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
torch::Tensor box_2d_iou(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
torch::Tensor box_2d_iou_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
