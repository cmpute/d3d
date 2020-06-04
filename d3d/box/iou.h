#pragma once

#include <torch/extension.h>
#include "d3d/box/common.h"

torch::Tensor iou2d(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const IouType iou_type
);
torch::Tensor iou2d_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const IouType iou_type
);
