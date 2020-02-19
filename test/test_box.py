import unittest

import torch
from d3d.box import rbox_2d_iou

class TestVoxelModule(unittest.TestCase):
    def test_iou(self):
        boxes1 = torch.tensor([
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [3, 3, 2, 2, 0]
        ], dtype=torch.float)
        boxes2 = torch.tensor([
            [3, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [1, 3, 2, 2, 0]
        ], dtype=torch.float)

        ious = rbox_2d_iou(boxes1, boxes2)
        ious_expected = torch.tensor([
            [0, 1, 0],
            [1, 4, 1],
            [0, 1, 0]
        ], dtype=torch.float)
        assert torch.allclose(ious, ious_expected)
