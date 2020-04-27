#pragma once

#include <torch/extension.h>

enum class ReductionType : int { NONE=0, MEAN=1, MAX=2, MIN=3 };
enum class MaxPointsFilterType : int { NONE=0, TRIM=1, FARTHEST_SAMPLING=2 };
enum class MaxVoxelsFilterType : int { NONE=0, TRIM=1, DESCENDING=2 };

py::dict voxelize_3d_dense(
    const torch::Tensor points, const torch::Tensor voxel_shape, const torch::Tensor voxel_bound,
    const int max_points, const int max_voxels, const ReductionType reduction_type
);

py::dict voxelize_sparse(
    const torch::Tensor points, const torch::Tensor voxel_size,
    const torch::optional<int> ndim
);

py::dict voxelize_filter(
    const torch::Tensor feats, const torch::Tensor points_mapping,
    const torch::Tensor coords, const torch::Tensor voxel_npoints, const torch::optional<torch::Tensor> coords_bound,
    const torch::optional<int> min_points, const torch::optional<int> max_points, const torch::optional<int> max_voxels,
    const torch::optional<MaxPointsFilterType> max_points_filter,
    const torch::optional<MaxVoxelsFilterType> max_voxels_filter
);
