#include <torch/extension.h>

#include "d3d/box/iou.h"
#include "d3d/box/nms.h"
#include "d3d/box/utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("box_2d_iou", &box_2d_iou, "IoU of 2D boxes");
    m.def("box_2d_iou_cuda", &box_2d_iou_cuda, "IoU of 2D boxes (using CUDA)");
    m.def("box_2d_nms", &box_2d_nms, "NMS on 2D boxes with sorted order");
    m.def("box_2d_nms_cuda", &box_2d_nms_cuda, "NMS on 2D boxes with sorted order (using CUDA)");
    m.def("rbox_2d_iou", &rbox_2d_iou, "IoU of 2D boxes with rotation");
    m.def("rbox_2d_iou_cuda", &rbox_2d_iou_cuda, "IoU of 2D boxes with rotation (using CUDA)");
    m.def("rbox_2d_nms", &rbox_2d_nms, "NMS on rotated 2D boxes with sorted order");
    m.def("rbox_2d_nms_cuda", &rbox_2d_nms_cuda, "NMS on rotated 2D boxes with sorted order (using CUDA)");
    m.def("rbox_2d_crop", &rbox_2d_crop, "Crop points from a point cloud within boxes");
}
