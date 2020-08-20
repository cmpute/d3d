
#include "d3d/box/iou.h"
#include "d3d/box/utils.h"
#include "d3d/geometry.hpp"
#include "d3d/geometry_grad.hpp"

using namespace std;
using namespace torch;
using namespace d3d;

template <typename scalar_t>
void iou2d_forward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_
) {
    using BoxType = AABox2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            ious_[i][j] = iou(bi, bj);
        }
    });
}

Tensor iou2d_forward(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2d_forward", [&] {
        iou2d_forward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2));
    });
    return ious;
}

template <typename scalar_t>
void iou2d_backward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    const _CpuAccessor(2) grad_,
    _CpuAccessor(2) grad_boxes1_,
    _CpuAccessor(2) grad_boxes2_
) {
    using BoxType = AABox2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            BoxType grad_i, grad_j;
            iou_grad(bi, bj, grad_[i][j], grad_i, grad_j);
             _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
             _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
        }
    });
}

void iou2d_backward(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    Tensor grad_boxes1, Tensor grad_boxes2
) {
    grad_boxes1 = torch::zeros_like(boxes1);
    grad_boxes2 = torch::zeros_like(boxes2);
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2d_backward", [&] {
        iou2d_backward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            grad._cpu_accessor(2),
            grad_boxes1._cpu_accessor(2),
            grad_boxes2._cpu_accessor(2));
    });
}
