import unittest

import numpy as np
import torch
from d3d.box import box2d_iou, box2d_nms, box2d_crop

sq2 = np.sqrt(2)
d90 = np.pi / 4

class TestVoxelModule(unittest.TestCase):
    def test_iou_aa_boxes(self):
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
        ious_expected = torch.tensor([
            [0,   1/7, 0  ],
            [1/7, 1,   1/7],
            [0,   1/7, 0  ]
        ], dtype=torch.float)

        ious = box2d_iou(boxes1, boxes2, method="box")
        assert torch.allclose(ious, ious_expected)

        ious = box2d_iou(boxes1.cuda(), boxes2.cuda(), method="box")
        assert torch.allclose(ious, ious_expected.cuda())

        ious = box2d_iou(boxes1, boxes2, method="rbox")
        assert torch.allclose(ious, ious_expected)

        ious = box2d_iou(boxes1.cuda(), boxes2.cuda(), method="rbox")
        assert torch.allclose(ious, ious_expected.cuda())

    def test_iou_rotated_boxes(self):
        boxes1 = torch.tensor([
            [0, 0, 2, 2, 0],
            [-1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0]
        ], dtype=torch.float)
        boxes2 = torch.tensor([
            [-1, 1, 2*sq2, 2*sq2, d90],
            [1, 1, sq2, sq2, d90]
        ], dtype=torch.float)
        
        rbox_ious_expected = torch.tensor([
            [1/5,  1/11],
            [1/2,  0],
            [1/11, 1/2]
        ], dtype=torch.float)
        box_ious_expected = torch.tensor([
            [1/4,  1/7],
            [1/4,  0],
            [1/9,  1]
        ], dtype=torch.float)

        ious = box2d_iou(boxes1, boxes2, method="box")
        assert torch.allclose(ious, box_ious_expected)

        ious = box2d_iou(boxes1.cuda(), boxes2.cuda(), method="box")
        assert torch.allclose(ious, box_ious_expected.cuda())

        ious = box2d_iou(boxes1, boxes2, method="rbox")
        assert torch.allclose(ious, rbox_ious_expected)

        ious = box2d_iou(boxes1.cuda(), boxes2.cuda(), method="rbox")
        assert torch.allclose(ious, rbox_ious_expected.cuda())

    def test_iou_apart_boxes(self):
        boxes = torch.tensor([
            [ 1,  2, 3, 3, 0],
            [-2,  1, 3, 3, 0],
            [-1, -2, 3, 3, 0],
            [ 2, -1, 3, 3, 0]
        ], dtype=torch.float)

        ious = box2d_iou(boxes, boxes, method="box")
        assert np.allclose(ious.numpy() - np.eye(4), 0, atol=1e-6)

        ious = box2d_iou(boxes.cuda(), boxes.cuda(), method="box")
        assert np.allclose(ious.cpu().numpy() - np.eye(4), 0, atol=1e-6)

        boxes = torch.tensor([
            [0, 0, 2, 2, 0],
            [ 2,  2, 2*sq2, 2*sq2, d90],
            [-2,  2, 2*sq2, 2*sq2, d90],
            [ 2, -2, 2*sq2, 2*sq2, d90],
            [-2, -2, 2*sq2, 2*sq2, d90]
        ], dtype=torch.float)

        ious = box2d_iou(boxes, boxes, method="rbox")
        assert np.allclose(ious.numpy() - np.eye(5), 0, atol=1e-6)

        ious = box2d_iou(boxes.cuda(), boxes.cuda(), method="rbox")
        assert np.allclose(ious.cpu().numpy() - np.eye(5), 0, atol=1e-6)

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
        mask_expected = torch.tensor([
            True, False, True, True, False, True
        ])

        mask = box2d_nms(boxes, scores)
        assert torch.all(mask == mask_expected)

        mask = box2d_nms(boxes.cuda(), scores.cuda())
        assert torch.all(mask == mask_expected.cuda())

        mask = box2d_nms(boxes, scores, method="rbox")
        assert torch.all(mask == mask_expected)

        mask = box2d_nms(boxes.cuda(), scores.cuda(), method="rbox")
        assert torch.all(mask == mask_expected.cuda())

    def test_box_crop(self):
        cloud = torch.rand(100,2) * 2 - 1
        boxes = torch.tensor([
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, d90]],
            dtype=torch.float)
        
        result = box2d_crop(cloud, boxes)
        abs_cloud = torch.abs(cloud)
        exp_box1, = torch.where(torch.all(abs_cloud < 0.5, 1))
        exp_box2, = torch.where(torch.abs(abs_cloud[:,0] + abs_cloud[:,1]) < sq2/2)

        assert len(result) == 2
        assert torch.all(result[0] == exp_box1)
        assert torch.all(result[1] == exp_box2)

if __name__ == "__main__":
    TestVoxelModule().test_nms()
