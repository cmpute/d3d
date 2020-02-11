import unittest

import torch

from d3d.utils.voxel import VoxelGenerator


class TestVoxelModule(unittest.TestCase):
    def test_generate_voxel(self):
        cloud = torch.rand((2000, 3), dtype=torch.float32)
        gen = VoxelGenerator([0,1, 0,1, 0,1], [10,10,10], 5)
        ret = gen(cloud)
