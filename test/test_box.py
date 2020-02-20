import unittest

import torch
from d3d.box import rbox_2d_iou, rbox_2d_nms

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
            [0,   1/7, 0  ],
            [1/7, 1,   1/7],
            [0,   1/7, 0  ]
        ], dtype=torch.float)
        assert torch.allclose(ious, ious_expected)

    def test_iou_cuda(self):
        device = torch.device('cuda')
        boxes1 = torch.tensor([
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [3, 3, 2, 2, 0]
        ], dtype=torch.float, device=device)
        boxes2 = torch.tensor([
            [3, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [1, 3, 2, 2, 0]
        ], dtype=torch.float, device=device)

        ious = rbox_2d_iou(boxes1, boxes2)
        assert ious.is_cuda
        ious_expected = torch.tensor([
            [0,   1/7, 0  ],
            [1/7, 1,   1/7],
            [0,   1/7, 0  ]
        ], dtype=torch.float, device=device)
        assert torch.allclose(ious, ious_expected)

    def test_nms(self):
        boxes = torch.tensor([
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [3, 3, 2, 2, 0],
            [3, 1, 1, 1, 0],
            [4, 2, 1, 1, 0],
            [5, 3, 1, 1, 0]
        ], dtype=torch.float)
        scores = torch.tensor([
            0.5, 0.3, 0.4, 0.4, 0.2, 0.1
        ], dtype=torch.float)

        mask = rbox_2d_nms(boxes, scores)

        mask_expected = torch.tensor([
            True, False, True, True, False, True
        ])
        assert torch.all(mask == mask_expected)

    def test_nms_cuda(self):
        device = torch.device('cuda')
        boxes = torch.tensor([
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [3, 3, 2, 2, 0],
            [3, 1, 1, 1, 0],
            [4, 2, 1, 1, 0],
            [5, 3, 1, 1, 0]
        ], dtype=torch.float, device=device)
        scores = torch.tensor([
            0.5, 0.3, 0.4, 0.4, 0.2, 0.1
        ], dtype=torch.float, device=device)

        mask = rbox_2d_nms(boxes, scores)
        assert mask.is_cuda
        mask_expected = torch.tensor([
            True, False, True, True, False, True
        ], device=device)
        assert torch.all(mask == mask_expected)
