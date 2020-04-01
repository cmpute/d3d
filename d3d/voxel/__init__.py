import torch
from .voxel_impl import voxelize_3d, voxelize_3d_sparse
from addict import Dict as edict

class VoxelGenerator:
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
            ret = voxelize_3d(points, self._shape, self._bounds,
                self._max_points, self._max_voxels, self._reduction)

        return edict(ret)
