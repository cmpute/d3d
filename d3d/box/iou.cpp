#include <d3d/box/iou.h>
#include <d3d/box/geometry.hpp>

using namespace torch;

void rbox_2d_iou(
    const Tensor boxes1, const Tensor boxes2, Tensor ious
)
{
    auto boxes1_ = boxes1.accessor<float, 2>();
    auto boxes2_ = boxes2.accessor<float, 2>();
    auto ious_ = ious.accessor<float, 2>();

    auto N = boxes1_.size(0);
    auto M = boxes2_.size(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
        {
            Box2 b1(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
                boxes1_[i][3], boxes1_[i][4]);
            Box2 b2(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
                boxes2_[j][3], boxes2_[j][4]);
            ious_[i][j] = b1.intersect(b2).area();
        }
}

void box_2d_iou(
    const Tensor boxes1, const Tensor boxes2, Tensor ious
)
{

}
