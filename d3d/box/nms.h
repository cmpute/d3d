#pragma once

#include <torch/extension.h>
#include "d3d/box/common.h"

torch::Tensor nms2d(
    const torch::Tensor boxes, const torch::Tensor scores,
    const IouType iou_type, const SupressionType supression_type,
    const float iou_threshold, const float score_threshold, const float supression_param
);

#ifdef BUILD_WITH_CUDA
torch::Tensor nms2d_cuda(
    const torch::Tensor boxes, const torch::Tensor scores,
    const IouType iou_type, const SupressionType supression_type,
    const float iou_threshold, const float score_threshold, const float supression_param
);
#endif
