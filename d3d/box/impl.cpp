#include <torch/extension.h>
#include <d3d/box/iou.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rbox_2d_iou", &rbox_2d_iou, "IoU of 2D boxes with rotation");
    m.def("rbox_2d_iou_cuda", &rbox_2d_iou_cuda, "IoU of 2D boxes with rotation (using CUDA)");
}
