import unittest

import torch
import numpy as np

from d3d.voxel import VoxelGenerator


class TestVoxelModule(unittest.TestCase):

    def test_generate_voxel(self):
        cloud = torch.rand((2000, 4), dtype=torch.float32)
        cloud = torch.cat((cloud, torch.tensor([
            [-1, -1, -1, -100], [-2, -2, -2, 100]
        ], dtype=torch.float32)), axis=0) # outlier point
        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10],
            reduction="mean", max_points=5, max_voxels=20000,
            max_points_filter="trim", max_voxels_filter="trim", dense=True)
        data = gen(cloud)
        assert len(data.voxels) == len(data.coords)
        assert len(data.voxels) <= 1000
        assert torch.all((data.voxels >= 0) & (data.voxels <= 1))
        assert torch.all((data.coords >= 0) & (data.coords <= 10))

        # ensure coordinate is correct
        for i in range(len(data.voxels)):
            for j in range(min(data.voxel_npoints[i], 5)):
                for k in range(3):
                    assert data.coords[i, k] == int(data.voxels[i, j, k] * 10)

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10],
            reduction="none", max_points=5, max_voxels=20000,
            max_points_filter="trim", max_voxels_filter="trim", dense=True)
        data = gen(cloud)
        assert len(data.voxels) == len(data.coords)
        assert 'aggregates' not in data
        assert len(data.voxels) <= 1000
        assert torch.all((data.voxels >= 0) & (data.voxels <= 1))
        assert torch.all((data.coords >= 0) & (data.coords <= 10))

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10])
        data = gen(cloud)
        assert len(data.points) == 2000
        assert len(data.coords) <= 1000
        assert torch.all((data.points >= 0) & (data.points <= 1))
        assert torch.all((data.coords >= 0) & (data.coords <= 10))

        for i in range(len(data.points)):
            vid = data.points_mapping[i]
            for k in range(3):
                assert data.coords[vid, k] == int(cloud[i, k] * 10)

    def test_filter(self):
        # test boundary filter
        cloud = torch.rand((2000, 3), dtype=torch.float32)
        cloud = (cloud - 0.5) * 4

        gen = VoxelGenerator([-1,1, -1,1, -1,1], [20,20,20])
        data = gen(cloud)
        assert torch.all((data.points >= -1) & (data.points <= 1))
        assert torch.all((data.coords >= 0) & (data.coords <= 20))

        for i in range(len(data.points)):
            vid = data.points_mapping[i]
            for k in range(3):
                assert data.coords[vid, k] == int((data.points[i, k]+1) * 10)

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], max_voxels=10, max_voxels_filter="trim")
        data = gen(cloud)
        assert len(data.coords) <= 10

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], max_voxels=10, max_voxels_filter="descending")
        data = gen(cloud)
        assert len(data.coords) <= 10

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], min_points=2, max_points=4, max_points_filter="trim")
        data = gen(cloud)
        assert torch.all((data.voxel_npoints >= 2) & (data.voxel_npoints <= 4))

    def test_generate_voxel_with_spconv(self):
        # only test dense representation
        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10],
            max_points=5, max_points_filter="trim", dense=True)
        data = np.load("test/voxel_data.npz") # generated using spconv.utils.VoxelGeneratorV2
        ret = gen(torch.tensor(data['cloud']))

        assert np.allclose(ret.voxels.numpy(), data['voxels'])
        assert np.allclose(ret.coords.numpy(), data['coords'])

if __name__ == "__main__":
    TestVoxelModule().test_generate_voxel()
