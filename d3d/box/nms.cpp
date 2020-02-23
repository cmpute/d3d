#include <d3d/box/nms.h>
#include <d3d/box/geometry.hpp>

using namespace std;

void rbox_2d_nms(
    const torch::Tensor boxes, const torch::Tensor order,
    float threshold,
    torch::Tensor suppressed
) {
    auto boxes_ = boxes.accessor<float, 2>();
    auto suppressed_ = suppressed.accessor<bool, 1>();
    auto order_ = order.accessor<long, 1>();

    int N = boxes_.size(0);
    for (int _i = 0; _i < N; _i++)
    {
        int i = order_[_i];
        if (suppressed_[i])
            continue;
        for (int _j = _i + 1; _j < N; _j++)
        {
            int j = order_[_j];
            if (suppressed_[j])
                continue;

            Box2 bi(boxes_[i][0], boxes_[i][1], boxes_[i][2],
                boxes_[i][3], boxes_[i][4]);
            Box2 bj(boxes_[j][0], boxes_[j][1], boxes_[j][2],
                boxes_[j][3], boxes_[j][4]);
            if (bi.iou(bj) > threshold)
                suppressed_[j] = true;
        }
    }
}
