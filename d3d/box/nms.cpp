#include <cmath>
#include "d3d/box/nms.h"
#include "d3d/box/utils.h"

using namespace std;
using namespace torch;

template <typename scalar_t, IouType Iou, SupressionType Supression>
void nms2d_templated(
    const _CpuAccessor(2) boxes_,
    _CpuAccessorT(long, 1) order_, // score order
    _CpuAccessor(1) scores_, // original score array
    const float iou_threshold,
    const float score_threshold,
    const float supression_param, // parameter for supression
    _CpuAccessorT(bool, 1) suppressed_
) {
    using BoxType = typename std::conditional<Iou == IouType::BOX, AABox2f, Box2f>::type;
    const int N = boxes_.size(0);

    // remove box under score threshold
    for (int _i = N - 1; _i > 0; _i--)
    {
        int i = order_[_i];
        if (scores_[i] > score_threshold)
            break;
        suppressed_[i] = true;
    }

    // main loop
    for (int _i = 0; _i < N; _i++)
    {
        int i = order_[_i];
        if (suppressed_[i])
        {
            if (Supression == SupressionType::HARD)
                continue;
            else break; // for soft-nms, remaining part are all suppressed
        }

        BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes_[i]);
        // suppress following boxes with lower score
        for (int _j = _i + 1; _j < N; _j++)
        {
            int j = order_[_j];
            if (Supression == SupressionType::HARD && suppressed_[j])
                continue;

            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes_[j]);
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
                    suppressed_[j] = scores_[j] < score_threshold;
                    break;

                case SupressionType::GAUSSIAN:
                    scores_[j] *= exp(-iou*iou/supression_param);
                    suppressed_[j] = scores_[j] < score_threshold;
                    break;
                }
            }
        }

        // For soft-NMS, we need to maintain score order
        if (Supression != SupressionType::HARD)
        {
            // find suppression block start
            int S = N - 1;
            while (S > _i && !suppressed_[order_[S]]) S--;

            // sort the scores again with simple insertion sort and put suppressed indices into back
            for (int _j = S - 1; _j > _i; _j--)
            {
                int j = order_[_j];
                int _k = _j + 1;
                while (_k < S && (suppressed_[j] || // suppression is like score = 0
                    scores_[order_[_k]] > scores_[j]))
                {
                    order_[_k-1] = order_[_k];
                    _k++;
                }
                order_[_k - 1] = j;
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
