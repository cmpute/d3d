#include "d3d/box/iou.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

Tensor box_2d_iou(
    const Tensor boxes1, const Tensor boxes2
) {
    auto boxes1_ = boxes1.accessor<float, 2>();
    auto boxes2_ = boxes2.accessor<float, 2>();
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, torch::dtype(torch::kFloat32));
    auto ious_ = ious.accessor<float, 2>();

    auto N = boxes1_.size(0);
    auto M = boxes2_.size(0);
    for (int i = 0; i < N; i++) // TODO: use parallel for // TODO: dispatch tensor types
    {
        AABox2 bi = Box2(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
            boxes1_[i][3], boxes1_[i][4]).bbox();
        for (int j = 0; j < M; j++)
        {
            AABox2 bj = Box2(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
                boxes2_[j][3], boxes2_[j][4]).bbox();
            ious_[i][j] = bi.iou(bj);
        }
    }

    return ious;
}

Tensor rbox_2d_iou(
    const Tensor boxes1, const Tensor boxes2
) {
    auto boxes1_ = boxes1.accessor<float, 2>();
    auto boxes2_ = boxes2.accessor<float, 2>();
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, torch::dtype(torch::kFloat32));
    auto ious_ = ious.accessor<float, 2>();

    auto N = boxes1_.size(0);
    auto M = boxes2_.size(0);
    for (int i = 0; i < N; i++)
    {
        Box2 bi(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
            boxes1_[i][3], boxes1_[i][4]);
        for (int j = 0; j < M; j++)
        {
            Box2 bj(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
                boxes2_[j][3], boxes2_[j][4]);
            ious_[i][j] = bi.iou(bj);
        }
    }

    return ious;
}
