import unittest

import torch
import numpy as np

from d3d.voxel import VoxelGenerator
from spconv.utils import VoxelGeneratorV2


class TestVoxelModule(unittest.TestCase):
    def test_generate_voxel(self):
        cloud = torch.rand((2000, 4), dtype=torch.float32)
        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5)
        data = gen(cloud)
        assert len(data.voxels) == len(data.coords)
        assert len(data.voxels) <= 1000

        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5, reduction="none")
        data = gen(cloud)
        assert len(data.voxels) == len(data.coords)
        assert 'aggregates' not in data
        assert len(data.voxels) <= 1000

    def test_generate_voxel_with_spconv(self):
        g1 = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5)
        g2 = VoxelGeneratorV2([0.1, 0.1, 0.1], [0,0,0, 1,1,1], 5)

        cloud = torch.rand((1000, 4), dtype=torch.float32)
        ret1 = g1(cloud)
        ret2 = g2.generate(cloud.numpy())

        assert np.allclose(ret1.voxels.numpy(), ret2['voxels'])
        assert np.allclose(ret1.coords.flip(1).numpy(), ret2['coordinates'])

if __name__ == "__main__":
    TestVoxelModule().test_generate_voxel_with_spconv()
