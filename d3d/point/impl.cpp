#include <torch/extension.h>

#include "d3d/point/scatter.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    #ifdef BUILD_WITH_CUDA
    m.attr("cuda_available") = py::bool_(true);
    #else
    m.attr("cuda_available") = py::bool_(false);
    #endif

    m.def("aligned_scatter_forward", &aligned_scatter_forward);
    m.def("aligned_scatter_backward", &aligned_scatter_backward);

    #ifdef BUILD_WITH_CUDA
    m.def("aligned_scatter_forward_cuda", &aligned_scatter_forward_cuda);
    m.def("aligned_scatter_backward_cuda", &aligned_scatter_backward_cuda);
    #endif

    py::enum_<AlignType>(m, "AlignType")
        .value("DROP", AlignType::DROP)
        .value("MEAN", AlignType::MEAN)
        .value("LINEAR", AlignType::LINEAR)
        .value("MAX", AlignType::MAX)
        .value("NEAREST", AlignType::NEAREST);
}
