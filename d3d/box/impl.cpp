#include <torch/extension.h>

#include <d3d/box/iou.h>
#include <d3d/box/nms.h>

using namespace std;
using namespace torch;

#include <d3d/box/geometry.hpp>

void rbox_2d_crop(Tensor cloud, Tensor boxes, py::list result)
{
    auto cloud_ = cloud.accessor<float, 2>();
    auto boxes_ = boxes.accessor<float, 2>();

    for (int i = 0; i < boxes_.size(0); i++)
    {
        Box2 box(boxes_[i][0], boxes_[i][1], boxes_[i][2],
            boxes_[i][3], boxes_[i][4]);
        AABox2 aabox = box.bbox();

        py::list box_result;
        for (int j = 0; j < cloud_.size(0); j++)
        {
            Point2 p (cloud_[j][0], cloud_[j][1]);
            if (aabox.contains(p) && box.contains(p))
                box_result.append(j);
        }
        result.append(box_result);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbox_2d_iou", &rbox_2d_iou, "IoU of 2D boxes with rotation");
    m.def("rbox_2d_iou_cuda", &rbox_2d_iou_cuda, "IoU of 2D boxes with rotation (using CUDA)");
    m.def("rbox_2d_nms", &rbox_2d_nms, "NMS on 2D boxes with sorted order");
    m.def("rbox_2d_nms_cuda", &rbox_2d_nms_cuda, "NMS on 2D boxes with sorted order (using CUDA)");
    m.def("rbox_2d_crop", &rbox_2d_crop, "Crop points from a point cloud within boxes");
}
