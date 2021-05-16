from addict import Dict as edict

try:
    import torch
    from .voxel_impl import (MaxPointsFilterType, MaxVoxelsFilterType,
                            ReductionType, voxelize_3d_dense, voxelize_3d_filter,
                            voxelize_3d_sparse)
except ImportError:
    raise ImportError("Cannot find compiled library! D3D is probably compiled without pytorch!")


class VoxelGenerator:
    '''
    Convert point cloud to voxels
    '''
    def __init__(self, bounds, shape,
        min_points=0, max_points=30, max_voxels=20000,
        max_points_filter=None, max_voxels_filter=None,
        reduction=None, dense=False):
        '''
        :param shape: The shape of the voxel grid
        :param bouds: The boundary of the voxel grid, in format [xmin, xmax, ymin, ymax, zmin, zmax]
        :param min_points: Minimum number of points per voxel
        :param max_points: Maximum number of points per voxel
        :param max_voxels: Maximum total number of voxels
        :param reduction: Type of reduction to apply, {none, mean, max, min}. Only for dense representation
        :param dense: Whether the output is in dense representation.
        :param max_voxels_filter: Filter to be applied to filter voxels
        :param max_points_filter: Filter to be applied to filter points in a voxel
        '''

        self._bounds = torch.tensor(bounds, dtype=torch.float)
        self._shape = torch.tensor(shape, dtype=torch.int32)
        self._min_points = min_points
        self._max_points = max_points
        self._max_voxels = max_voxels
        self._dense = dense

        # calculate tensors for sparse representation
        bounds_array = self._bounds.reshape(3, 2)
        self._size = (bounds_array[:, 1] - bounds_array[:, 0]) / self._shape
        bounds_dist = bounds_array[:, 0] / self._size
        if torch.any(torch.abs(torch.round(bounds_dist) - bounds_dist) > 1e-3):
            raise ValueError("The voxelization grids is not aligned with the origin, which could lead to unexpected behavior!")
        self._offset = torch.round(bounds_dist).int()
        self._vbounds = torch.round(bounds_array / self._size.reshape(3, 1)).long()

        # convert reduction type
        reduction = (reduction or "NONE").upper()
        if reduction != "NONE" and not dense:
            raise ValueError("Reduction is only for dense voxelization!")
        if reduction in ReductionType.__dict__:
            self._reduction = getattr(ReductionType, reduction)
        else:
            raise ValueError("Unsupported reduction type in VoxelGenerator!")

        # convert filter types
        max_points_filter = (max_points_filter or "NONE").upper()
        if max_points_filter in MaxPointsFilterType.__dict__:
            self._max_points_filter = getattr(MaxPointsFilterType, max_points_filter)
        else:
            raise ValueError("Unsupported maximum points filter in VoxelGenerator!")

        max_voxels_filter = (max_voxels_filter or "NONE").upper()
        if max_voxels_filter in MaxVoxelsFilterType.__dict__:
            self._max_voxels_filter = getattr(MaxVoxelsFilterType, max_voxels_filter)
        else:
            raise ValueError("Unsupported maximum voxels filter in VoxelGenerator!")

        # check filter implementation
        if dense:
            if min_points > 0:
                raise NotImplementedError("Minimum points filtering is not implemented for dense")
            if self._max_points_filter not in [MaxPointsFilterType.NONE, MaxPointsFilterType.TRIM]:
                raise NotImplementedError("Only trim is implemented for max points filtering")
            if self._max_voxels_filter not in [MaxVoxelsFilterType.NONE, MaxVoxelsFilterType.TRIM]:
                raise NotImplementedError("Only trim is implemented for max voxels filtering")

    def __call__(self, points):
        '''
        :param points: Original point cloud with certain features. The first three columns will be considered as xyz coordinates

        :return: A dictionary of data
            voxels: Generated features for each voxel (dense only)
            points: Filtered point cloud (sparse only)
            coords: Calculated coordinates for generated voxels
            aggregates: Aggregated feature of each voxel, only applicable when reduction_type != 0 (dense only)
            voxel_pmask: Point mask of each voxel (dense only)
            voxel_npoints: Actual number of points in each voxel (Note: this could be larger than max_npoints when dense)
            points_mapping: Map from points to voxel index
        '''
        if self._dense:
            ret = edict(voxelize_3d_dense(points, self._shape, self._bounds,
                self._max_points, self._max_voxels, self._reduction))
        else:
            sparse = edict(voxelize_3d_sparse(points, self._size, 3))
            ret = edict(voxelize_3d_filter(
                points, sparse.points_mapping,
                sparse.coords, sparse.voxel_npoints, self._vbounds,
                self._min_points, self._max_points, self._max_voxels,
                self._max_points_filter, self._max_voxels_filter
            ))
            ret.coords = ret.coords - self._offset
        return ret
