#include "d3d/voxel/voxelize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_3d", &voxelize_3d, "3D voxelization of tensor");
    m.def("voxelize_3d_sparse", &voxelize_3d_sparse, "3D voxelization of tensor in sparse point representation");
}
