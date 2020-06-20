#include <vector>
#include "d3d/box/utils.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

py::list rbox_2d_crop(Tensor cloud, Tensor boxes)
{
    py::list result;
    auto cloud_ = cloud.accessor<float, 2>();
    auto boxes_ = boxes.accessor<float, 2>();

    for (int i = 0; i < boxes_.size(0); i++)
    {
        Box2f box(boxes_[i][0], boxes_[i][1], boxes_[i][2],
            boxes_[i][3], boxes_[i][4]);
        AABox2f aabox = box.bbox();

        vector<int> box_result;
        for (int j = 0; j < cloud_.size(0); j++)
        {
            Point2f p(cloud_[j][0], cloud_[j][1]);
            if (aabox.contains(p) && box.contains(p))
                box_result.push_back(j);
        }
        result.append(torch::tensor(box_result, torch::kLong));
    }

    return result;
}