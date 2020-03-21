import unittest

import torch
import numpy as np

from d3d.voxel import VoxelGenerator


class TestVoxelModule(unittest.TestCase):
    def test_generate_voxel(self):
        cloud = torch.rand((2000, 4), dtype=torch.float32)
        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5)
        data = gen(cloud)
        assert len(data.voxels) == len(data.coords)
        assert len(data.voxels) <= 1000
        assert torch.all((data.coords >= 0) & (data.coords <= 10))

        # ensure coordinate is correct
        for i in range(len(data.voxels)):
            for j in range(min(data.voxel_npoints[i], 5)):
                for k in range(3):
                    assert data.coords[i, k] == int(data.voxels[i, j, k] * 10)

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5, reduction="none")
        data = gen(cloud)
        assert len(data.voxels) == len(data.coords)
        assert 'aggregates' not in data
        assert len(data.voxels) <= 1000
        assert torch.all((data.coords >= 0) & (data.coords <= 10))

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5, sparse_repr=True)
        data = gen(cloud)
        assert len(data.coords) <= 1000
        assert torch.all((data.coords >= 0) & (data.coords <= 10))

        for i in range(len(cloud)):
            vid = data.points_mapping[i]
            for k in range(3):
                assert data.coords[vid, k] == int(cloud[i, k] * 10)

    def test_generate_voxel_with_spconv(self):
        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5)
        data = np.load("test/voxel_data.npz") # generated using spconv.utils.VoxelGeneratorV2
        ret = gen(torch.tensor(data['cloud']))

        assert np.allclose(ret.voxels.numpy(), data['voxels'])
        assert np.allclose(ret.coords.numpy(), data['coords'])

if __name__ == "__main__":
    TestVoxelModule().test_generate_voxel_with_spconv()
