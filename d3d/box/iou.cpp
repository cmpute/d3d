#include "d3d/box/iou.h"
#include "d3d/box/utils.h"

using namespace std;
using namespace torch;

template <typename scalar_t, IouType Iou>
void iou2d_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_
) {
    using BoxType = typename std::conditional<Iou == IouType::BOX, AABox2f, Box2f>::type;
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
            ious_[i][j] = bi.iou(bj);
        }
    });
}

Tensor iou2d(
    const Tensor boxes1, const Tensor boxes2, const IouType iou_type
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2d", _NMS_DISPATCH_IOUTYPE(iou_type, [&] {
        iou2d_templated<scalar_t, Iou>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2));
    }));
    return ious;
}
