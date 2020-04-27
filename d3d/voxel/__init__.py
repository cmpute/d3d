import torch
from .voxel_impl import voxelize_3d_dense, voxelize_3d_sparse, ReductionType, MaxPointsFilterType, MaxVoxelsFilterType
from addict import Dict as edict

class VoxelGeneratorOld:
    # TODO: directly output tensors with type long
    # TODO: support max_points, max_voxels for sparse representation (default for max_voxels should be changed to -1 (not set)), and sampling strategy should be implemented
    # TODO: support min_points to filter out voxels with less points
    # TODO: bounds and shape are not actually necessary, we can generate infinite voxels in sparse representation, so the generator can be divided into BoundedVoxelGenerator and SparseVoxelGenerator
    def __init__(self, bounds, shape,
        max_points, max_voxels=20000, reduction="mean", sparse_repr=False):

        self._bounds = torch.tensor(bounds, dtype=torch.float)
        self._shape = torch.tensor(shape, dtype=torch.int32)
        self._max_points = max_points
        self._max_voxels = max_voxels
        self._sparse_repr = sparse_repr

        if reduction.lower() == "none":
            self._reduction = 0
        elif reduction.lower() == "mean":
            self._reduction = 1
        elif reduction.lower() == "max":
            self._reduction = 2
        elif reduction.lower() == "min":
            self._reduction = 3
        else:
            raise ValueError("Unsupported reduction type in VoxelGenerator!")

    def __call__(self, points):
        
        if self._sparse_repr:
            ret = voxelize_3d_sparse(points, self._shape, self._bounds,
                self._max_points, self._max_voxels)
        else:
            ret = voxelize_3d_dense(points, self._shape, self._bounds,
                self._max_points, self._max_voxels, self._reduction)

        return edict(ret)

class VoxelGenerator:
    def __init__(self, bounds, shape,
        min_points=0, max_points=30, max_voxels=20000,
        max_points_filter=None, max_voxels_filter=None,
        reduction=None, dense=False):

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
            self._max_voxels_filter = getattr(MaxVoxelsFilterType, max_points_filter)
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
        if self._dense:
            ret = edict(voxelize_3d_dense(points, self._shape, self._bounds,
                self._max_points, self._max_voxels, self._reduction))
        else:
            from .voxel_impl import voxelize_3d_sparse_coord, voxelize_3d_sparse_filter
            sparse = edict(voxelize_3d_sparse_coord(points, self._size, 3))
            ret = edict(voxelize_3d_sparse_filter(
                points, sparse.points_mapping,
                sparse.coords, sparse.voxel_npoints, self._vbounds,
                self._min_points, self._max_points, self._max_voxels,
                self._max_points_filter, self._max_voxels_filter
            ))
            ret.coords = ret.coords + self._offset
        return ret
