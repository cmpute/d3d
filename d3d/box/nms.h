#include <torch/extension.h>

void rbox_2d_nms(
    const torch::Tensor boxes, const torch::Tensor order,
    float threshold,
    torch::Tensor suppressed
);

void rbox_2d_nms_cuda(
    const torch::Tensor boxes, const torch::Tensor order,
    float threshold,
    torch::Tensor suppressed
);
