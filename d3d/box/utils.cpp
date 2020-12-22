#include <vector>
#include "d3d/box/utils.h"
#include "dgal/geometry.hpp"

using namespace std;
using namespace torch;
using namespace dgal;

template <typename scalar_t>
void crop_2dr_templated(
    const _CpuAccessor(2) cloud_,
    const _CpuAccessor(2) boxes_,
    _CpuAccessorT(bool, 2) indicators_
) {
    const auto N = cloud_.size(0);
    const auto M = boxes_.size(0);

    parallel_for(0, M, 1, [&](int64_t begin, int64_t end)
    {
        for (int64_t i = begin; i < end; i++)
        {
            Quad2<scalar_t> box = poly2_from_xywhr(boxes_[i][0], boxes_[i][1], boxes_[i][2],
                boxes_[i][3], boxes_[i][4]);
            AABox2<scalar_t> aabox = aabox2_from_poly2(box);

            for (int j = 0; j < N; j++)
            {
                Point2<scalar_t> p {.x=cloud_[j][0], .y=cloud_[j][1]};
                if (aabox.contains(p) && box.contains(p))
                    indicators_[i][j] = true;
                else
                    indicators_[i][j] = false;
            }
        }
    });
}

Tensor crop_2dr(const Tensor cloud, const Tensor boxes)
{
    Tensor indicators = torch::empty({boxes.size(0), cloud.size(0)}, torch::dtype(torch::kBool));
    AT_DISPATCH_FLOATING_TYPES(cloud.scalar_type(), "crop_2dr_templated", [&] {
        crop_2dr_templated<scalar_t>(
            cloud._cpu_accessor(2),
            boxes._cpu_accessor(2),
            indicators._cpu_accessor_t(bool, 2));
    });
    return indicators;
}