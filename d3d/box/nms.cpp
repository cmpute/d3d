#include "d3d/box/nms.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

template <typename scalar_t>
void box_2d_nms_templated(
    const _CpuAccessor(2) boxes_,
    const _CpuAccessorT(long, 1) order_,
    const float threshold,
    _CpuAccessorT(bool, 1) suppressed_
) {
    const int N = boxes_.size(0);
    for (int _i = 0; _i < N; _i++)
    {
        int i = order_[_i];
        if (suppressed_[i])
            continue;

        AABox2 bi = Box2(boxes_[i][0], boxes_[i][1], boxes_[i][2],
            boxes_[i][3], boxes_[i][4]).bbox();
        for (int _j = _i + 1; _j < N; _j++)
        {
            int j = order_[_j];
            if (suppressed_[j])
                continue;

            AABox2 bj = Box2(boxes_[j][0], boxes_[j][1], boxes_[j][2],
                boxes_[j][3], boxes_[j][4]).bbox();
            if (bi.iou(bj) > threshold)
                suppressed_[j] = true;
        }
    }
}

Tensor box_2d_nms(
    const Tensor boxes, const Tensor order, const float threshold
) {
    Tensor suppressed = torch::zeros({boxes.size(0)}, torch::dtype(torch::kBool));
    AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "box_2d_nms", ([&] {
        box_2d_nms_templated<scalar_t>(
            boxes._cpu_accessor(2),
            order._cpu_accessor_t(long, 1),
            threshold,
            suppressed._cpu_accessor_t(bool, 1));
    }));
    return suppressed;
}

template <typename scalar_t>
void rbox_2d_nms_templated(
    const _CpuAccessor(2) boxes_,
    const _CpuAccessorT(long, 1) order_,
    const float threshold,
    _CpuAccessorT(bool, 1) suppressed_
) {
    const int N = boxes_.size(0);
    for (int _i = 0; _i < N; _i++)
    {
        int i = order_[_i];
        if (suppressed_[i])
            continue;

        Box2 bi(boxes_[i][0], boxes_[i][1], boxes_[i][2],
            boxes_[i][3], boxes_[i][4]);
        for (int _j = _i + 1; _j < N; _j++)
        {
            int j = order_[_j];
            if (suppressed_[j])
                continue;

            Box2 bj(boxes_[j][0], boxes_[j][1], boxes_[j][2],
                boxes_[j][3], boxes_[j][4]);
            if (bi.iou(bj) > threshold)
                suppressed_[j] = true;
        }
    }
}

Tensor rbox_2d_nms(
    const Tensor boxes, const Tensor order, const float threshold
) {
    Tensor suppressed = torch::zeros({boxes.size(0)}, torch::dtype(torch::kBool));
    AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "rbox_2d_nms", ([&] {
        rbox_2d_nms_templated<scalar_t>(
            boxes._cpu_accessor(2),
            order._cpu_accessor_t(long, 1),
            threshold,
            suppressed._cpu_accessor_t(bool, 1));
    }));
    return suppressed;
}

