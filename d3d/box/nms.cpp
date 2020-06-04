#include <cmath>
#include "d3d/box/nms.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

// template utils
template <typename scalar_t, typename TBox>
struct _BoxUtil { static TBox make_box(const _CpuAccessor(1) data); };
template <typename scalar_t>
struct _BoxUtil<scalar_t, Poly2>
{
    static Poly2 make_box(const _CpuAccessor(1) data)
    {
        return make_box2(data[0], data[1], data[2], data[3], data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtil<scalar_t, AABox2>
{
    static AABox2 make_box(const _CpuAccessor(1) data)
    {
        return make_box2(data[0], data[1], data[2], data[3], data[4]).bbox();
    }
};

template <typename scalar_t, IouType Iou, SupressionType Supression>
void nms2d_templated(
    const _CpuAccessor(2) boxes_,
    const _CpuAccessorT(long, 1) order_, // score order
    _CpuAccessor(1) scores_, // original score array
    const float iou_threshold,
    const float score_threshold,
    const float supression_param, // parameter for supression
    _CpuAccessorT(bool, 1) suppressed_
) {
    using BoxType = typename std::conditional<Iou == IouType::BOX, AABox2, Poly2>::type;
    const int N = boxes_.size(0);

    // remove box under score threshold
    for (int i = 0; i < N; i++)
        if (scores_[i] < score_threshold)
            suppressed_[i] = true;

    // main loop
    for (int _i = 0; _i < N; _i++)
    {
        int i = order_[_i];
        if (suppressed_[i])
            continue;

        BoxType bi = _BoxUtil<scalar_t, BoxType>::make_box(boxes_[i]);
        // Supress following boxes with lower score
        for (int _j = _i + 1; _j < N; _j++)
        {
            int j = order_[_j];
            if (suppressed_[j])
                continue;

            BoxType bj = _BoxUtil<scalar_t, BoxType>::make_box(boxes_[j]);
            scalar_t iou = bi.iou(bj);

            if (iou > iou_threshold)
            {
                switch (Supression)
                {
                case SupressionType::HARD:
                        suppressed_[j] = true;
                    break;
                
                case SupressionType::LINEAR:
                    scores_[j] *= 1 - pow(iou, supression_param);
                    if (scores_[j] < score_threshold)
                        suppressed_[j] = true;
                    break;

                case SupressionType::GAUSSIAN:
                    scores_[j] *= exp(-iou*iou/supression_param);
                    if (scores_[j] < score_threshold)
                        suppressed_[j] = true;
                    break;
                }
            }
        }
    }
}

Tensor nms2d(
    const Tensor boxes, const Tensor scores,
    const IouType iou_type, const SupressionType supression_type,
    const float iou_threshold, const float score_threshold, const float supression_param
) {
    Tensor order = scores.argsort(-1, true);
    Tensor scores_copy = torch::empty_like(scores);
    scores_copy.copy_(scores);
    Tensor suppressed = torch::zeros({boxes.size(0)}, torch::dtype(torch::kBool));

    AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "nms2d",
        _NMS_DISPATCH_IOUTYPE(iou_type, _NMS_DISPATCH_SUPRESSTYPE(supression_type, [&] {
            nms2d_templated<scalar_t, Iou, Supression>(
                boxes._cpu_accessor(2),
                order._cpu_accessor_t(long, 1),
                scores_copy._cpu_accessor(1),
                iou_threshold, score_threshold, supression_param,
                suppressed._cpu_accessor_t(bool, 1));
        }))
    );

    return suppressed;
}
