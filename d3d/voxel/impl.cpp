#include "d3d/voxel/voxelize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_3d_dense", &voxelize_3d_dense, "3D voxelization of tensor");
    m.def("voxelize_3d_sparse", &voxelize_3d_sparse, "3D voxelization of tensor in sparse point representation");
    m.def("voxelize_3d_sparse_coord", &voxelize_sparse, "3D voxelization of point cloud");
    m.def("voxelize_3d_sparse_filter", &voxelize_filter, "Filter generated voxels");

    py::enum_<ReductionType>(m, "ReductionType")
        .value("NONE", ReductionType::NONE)
        .value("MEAN", ReductionType::MEAN)
        .value("MIN", ReductionType::MIN)
        .value("MAX", ReductionType::MAX);
    py::enum_<MaxPointsFilterType>(m, "MaxPointsFilterType")
        .value("NONE", MaxPointsFilterType::NONE)
        .value("TRIM", MaxPointsFilterType::TRIM)
        .value("FARTHEST_SAMPLING", MaxPointsFilterType::FARTHEST_SAMPLING);
    py::enum_<MaxVoxelsFilterType>(m, "MaxVoxelsFilterType")
        .value("NONE", MaxVoxelsFilterType::NONE)
        .value("TRIM", MaxVoxelsFilterType::TRIM)
        .value("DESCENDING", MaxVoxelsFilterType::DESCENDING);
}
