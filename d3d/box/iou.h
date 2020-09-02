#pragma once

#include <tuple>
#include <torch/extension.h>
#include "d3d/box/common.h"

torch::Tensor iou2d_forward(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> iou2d_backward(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad
);

std::tuple<torch::Tensor /*iou*/, torch::Tensor /*nx*/, torch::Tensor /*xflags*/> iou2dr_forward(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> iou2dr_backward(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    const torch::Tensor nx, const torch::Tensor xflags
);

std::tuple<torch::Tensor /*iou*/, torch::Tensor /*nx & nm*/, torch::Tensor /*xflags & mflags*/> giou2dr_forward(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> giou2dr_backward(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    const torch::Tensor nxm, const torch::Tensor xmflags
);

std::tuple<torch::Tensor /*iou*/, torch::Tensor /*nx & dflags*/, torch::Tensor /*xflags*/> diou2dr_forward(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> diou2dr_backward(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    const torch::Tensor nxd, const torch::Tensor xflags
);

#ifdef BUILD_WITH_CUDA
torch::Tensor iou2d_forward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> iou2d_backward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad
);

std::tuple<torch::Tensor /*iou*/, torch::Tensor /*nx*/, torch::Tensor /*xflags*/> iou2dr_forward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> iou2dr_backward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    const torch::Tensor nx, const torch::Tensor xflags
);

std::tuple<torch::Tensor /*iou*/, torch::Tensor /*nx & nm*/, torch::Tensor /*xflags & mflags*/> giou2dr_forward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> giou2dr_backward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    const torch::Tensor nxm, const torch::Tensor xmflags
);

std::tuple<torch::Tensor /*iou*/, torch::Tensor /*nx & dflags*/, torch::Tensor /*xflags*/> diou2dr_forward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2
);
std::tuple<torch::Tensor /*grad_boxes1*/, torch::Tensor /*grad_boxes2*/> diou2dr_backward_cuda(
    const torch::Tensor boxes1, const torch::Tensor boxes2, const torch::Tensor grad,
    const torch::Tensor nxd, const torch::Tensor xflags
);
#endif
