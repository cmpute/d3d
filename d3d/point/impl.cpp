#include <torch/extension.h>

#include "d3d/point/scatter.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aligned_scatter_forward", &aligned_scatter_forward);
    m.def("aligned_scatter_forward_cuda", &aligned_scatter_forward_cuda);

    py::enum_<AlignType>(m, "AlignType")
        .value("DROP", AlignType::DROP)
        .value("MEAN", AlignType::MEAN)
        .value("LINEAR", AlignType::LINEAR)
        .value("MAX", AlignType::MAX)
        .value("NEAREST", AlignType::NEAREST);
}
