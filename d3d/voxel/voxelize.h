#pragma once

#include <torch/extension.h>

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
py::dict voxelize_3d(
    const torch::Tensor points, const torch::Tensor voxel_shape, const torch::Tensor voxel_bound,
    const int max_points, const int max_voxels, const int reduction_type
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