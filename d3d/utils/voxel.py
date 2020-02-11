import torch
from ._impl import voxelize_3d
from addict import Dict as edict

class VoxelGenerator:
    def __init__(self, bounds, shape,
        max_points, max_voxels=20000, reduction="mean"):

        self._bounds = torch.tensor(bounds, dtype=torch.float)
        self._shape = torch.tensor(shape, dtype=torch.int32)
        self._voxel_size = None # TODO: calculate
        self._max_points = max_points
        self._max_voxels = max_voxels

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

        # initialize variables
        voxels = torch.empty((self._max_voxels, self._max_points, points.shape[-1]), dtype=torch.float32)
        coords = torch.empty((self._max_voxels, 3), dtype=torch.int32)
        aggregates = torch.empty((self._max_voxels, points.shape[-1]), dtype=torch.float32)\
            if self._reduction else torch.empty(0)
        voxel_pmask = torch.empty((self._max_voxels, self._max_points), dtype=torch.bool)
        voxel_npoints = torch.empty(self._max_voxels, dtype=torch.int32)

        # call implementation
        nvoxels = voxelize_3d(points, self._shape, self._bounds,
            self._max_voxels, self._max_voxels, self._reduction,
            voxels, coords, aggregates, voxel_pmask, voxel_npoints
        )
        
        return edict(
            voxels = voxels[:nvoxels],
            coords = coords[:nvoxels],
            aggregates = aggregates[:nvoxels],
            voxel_pmask = voxel_pmask[:nvoxels],
            voxel_npoints = voxel_npoints[:nvoxels],
        )

