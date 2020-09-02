#include <torch/extension.h>

#include "d3d/box/iou.h"
#include "d3d/box/nms.h"
#include "d3d/box/utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    #ifdef BUILD_WITH_CUDA
    m.attr("cuda_available") = py::bool_(true);
    #else
    m.attr("cuda_available") = py::bool_(false);
    #endif

    m.def("iou2d_forward", &iou2d_forward, "IoU of 2D boxes");
    m.def("iou2d_backward", &iou2d_backward);
    m.def("iou2dr_forward", &iou2dr_forward, "Rotated IoU of 2D boxes");
    m.def("iou2dr_backward", &iou2dr_backward);
    m.def("giou2dr_forward", &giou2dr_forward, "Rotated GIoU of 2D boxes");
    m.def("giou2dr_backward", &giou2dr_backward);
    m.def("diou2dr_forward", &diou2dr_forward, "Rotated GIoU of 2D boxes");
    m.def("diou2dr_backward", &diou2dr_backward);
    m.def("nms2d", &nms2d, "NMS on 2D boxes");
    m.def("rbox_2d_crop", &rbox_2d_crop, "Crop points from a point cloud within boxes");

    #ifdef BUILD_WITH_CUDA
    m.def("iou2d_forward_cuda", &iou2d_forward_cuda, "IoU of 2D boxes (using CUDA)");
    m.def("iou2d_backward_cuda", &iou2d_backward_cuda);
    m.def("iou2dr_forward_cuda", &iou2dr_forward_cuda, "Rotated IoU of 2D boxes (using CUDA)");
    m.def("iou2dr_backward_cuda", &iou2dr_backward_cuda);
    m.def("giou2dr_forward_cuda", &giou2dr_forward_cuda, "Rotated GIoU of 2D boxes (using CUDA)");
    m.def("giou2dr_backward_cuda", &giou2dr_backward_cuda);
    m.def("diou2dr_forward_cuda", &diou2dr_forward_cuda, "Rotated GIoU of 2D boxes (using CUDA)");
    m.def("diou2dr_backward_cuda", &diou2dr_backward_cuda);
    m.def("nms2d_cuda", &nms2d_cuda, "NMS on 2D boxes (using CUDA)");
    #endif
    
    py::enum_<IouType>(m, "IouType")
        .value("NA", IouType::NA)
        .value("BOX", IouType::BOX)
        .value("RBOX", IouType::RBOX)
        .value("GBOX", IouType::GBOX)
        .value("GRBOX", IouType::GRBOX)
        .value("DBOX", IouType::DBOX)
        .value("DRBOX", IouType::DRBOX);
    py::enum_<SupressionType>(m, "SupressionType")
        .value("HARD", SupressionType::HARD)
        .value("LINEAR", SupressionType::LINEAR)
        .value("GAUSSIAN", SupressionType::GAUSSIAN);
}
