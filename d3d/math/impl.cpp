#include <torch/extension.h>
#include "d3d/math/math.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("i0e", &i0e, "Exponentially scaled modified Bessel function of order 0");
    m.def("i0e_cuda", &i0e_cuda, "Exponentially scaled modified Bessel function of order 0");
    m.def("i1e", &i1e, "Exponentially scaled modified Bessel function of order 1");
    m.def("i1e_cuda", &i1e_cuda, "Exponentially scaled modified Bessel function of order 1");
}
