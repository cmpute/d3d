#pragma once

#include <torch/extension.h>

enum class ReductionType : int { NONE=0, MEAN=1, MAX=2, MIN=3 };
enum class MaxPointsFilterType : int { NONE=0, TRIM=1, FARTHEST_SAMPLING=2 };
enum class MaxVoxelsFilterType : int { NONE=0, TRIM=1, DESCENDING=2 };

/*
 * This function voxelize point cloud
 * 
 * \param[in] points Original point cloud with certain features. The first three columns have to be xyz coordinates
 * \param[in] voxel_shape The shape of the voxel grid
 * \param[in] voxel_bound The boundary of the voxel grid, in format [xmin, xmax, ymin, ymax, zmin, zmax]
 * \param[in] max_points Maximum number of points per voxel
 * \param[in] max_voxels Maximum number of voxels
 * \param[in] reduction_type type of reduction to apply, {0:none, 1:mean, 2:max, 3:min}
 * \param[out] voxels Generated features for each voxel
 * \param[out] coords Calculated coordinates for generated voxels
 * \param[out] aggregates Aggregated feature of each voxel, only applicable when reduction_type != 0
 * \param[out] voxel_pmask Point mask of each voxel
 * \param[out] voxel_npoints Number of points in each voxel (Note: this could be larger than max_npoints)
 */
py::dict voxelize_3d_dense(
    const torch::Tensor points, const torch::Tensor voxel_shape, const torch::Tensor voxel_bound,
    const int max_points, const int max_voxels, const ReductionType reduction_type
);

/*
 * This function has same functionality as voxelize_3d, but it returns sparse representation of points without discard and reduction.
 * \param[out] coords Calculated coordinates for generated voxels
 * \param[out] points_mapping Each index of voxel that a point belongs to.
 * \param[out] voxel_npoints Number of points in each voxel
 */
py::dict voxelize_3d_sparse(
    const torch::Tensor points, const torch::Tensor voxel_shape, const torch::Tensor voxel_bound,
    const torch::optional<int> max_points, const torch::optional<int> max_voxels
);

py::dict voxelize_sparse(
    const torch::Tensor points, const torch::Tensor voxel_size,
    const torch::optional<int> ndim
);

/*
 * \param[in] densify whether convert the voxel representation to dense tensor. If densify is true, then max_points must be specified.
 *     If max_voxels is specified, then then tensor shape will be max_voxels * max_points * features.
 */
py::dict voxelize_filter(
    const torch::Tensor feats, const torch::Tensor points_mapping,
    const torch::Tensor coords, const torch::Tensor voxel_npoints, const torch::optional<torch::Tensor> coords_bound,
    const torch::optional<int> min_points, const torch::optional<int> max_points, const torch::optional<int> max_voxels,
    const torch::optional<MaxPointsFilterType> max_points_filter,
    const torch::optional<MaxVoxelsFilterType> max_voxels_filter
);
