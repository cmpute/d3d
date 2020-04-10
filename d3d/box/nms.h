#include <torch/extension.h>

torch::Tensor box_2d_nms(
    const torch::Tensor boxes, const torch::Tensor order, const float threshold
);
torch::Tensor rbox_2d_nms(
    const torch::Tensor boxes, const torch::Tensor order, const float threshold
);

torch::Tensor box_2d_nms_cuda(
    const torch::Tensor boxes, const torch::Tensor order, const float threshold
);
torch::Tensor rbox_2d_nms_cuda(
    const torch::Tensor boxes, const torch::Tensor order, const float threshold
);

