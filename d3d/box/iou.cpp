#include "d3d/box/iou.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

template <typename scalar_t>
void box_2d_iou_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_
) {
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);
    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++) // TODO: use parallel for // TODO: dispatch tensor types
        {
            const int i = nm % N;
            const int j = nm / N;

            AABox2 bi = Box2(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
                boxes1_[i][3], boxes1_[i][4]).bbox();
            AABox2 bj = Box2(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
                boxes2_[j][3], boxes2_[j][4]).bbox();
            ious_[i][j] = bi.iou(bj);
        }
    });
}

Tensor box_2d_iou(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "box_2d_iou", ([&] {
        box_2d_iou_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2));
    }));
    return ious;
}

template <typename scalar_t>
void rbox_2d_iou_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_
) {
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);
    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;
            Box2 bi(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
                boxes1_[i][3], boxes1_[i][4]);
            Box2 bj(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
                boxes2_[j][3], boxes2_[j][4]);
            ious_[i][j] = bi.iou(bj);
        }
    });
}

Tensor rbox_2d_iou(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "rbox_2d_iou", ([&] {
        rbox_2d_iou_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2));
    }));
    return ious;
}
