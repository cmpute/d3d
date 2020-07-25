import unittest
import numpy as np

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import KittiObjectClass
from d3d.abstraction import ObjectTarget3DArray, ObjectTarget3D, ObjectTag
from d3d.tracking.matcher import DistanceTypes, NearestNeighborMatcher

class TestMatchers(unittest.TestCase):
    def test_nearest_neighbor_matcher(self):
        matcher = NearestNeighborMatcher()

        # src boxes
        r = Rotation.from_euler("Z", 0)
        d = [2, 2, 2]
        dt1 = ObjectTarget3D(
            [0, 0, 0], r, d,
            ObjectTag(KittiObjectClass.Car, scores=0.8)
        )
        dt2 = ObjectTarget3D(
            [1, 1, 1], r, d,
            ObjectTag(KittiObjectClass.Van, scores=0.7)
        )
        dt3 = ObjectTarget3D(
            [-1, -1, -1], r, d,
            ObjectTag(KittiObjectClass.Pedestrian, scores=0.8)
        )
        dt_boxes = ObjectTarget3DArray([dt1, dt2, dt3], frame="test")

        # dst boxes
        r = Rotation.from_euler("Z", 0.01)
        d = [2.1, 2.1, 2.1]
        gt1 = ObjectTarget3D(
            [0, 0, 0], r, d,
            ObjectTag(KittiObjectClass.Van)
        )
        gt2 = ObjectTarget3D(
            [-1, 1, 0], r, d,
            ObjectTag(KittiObjectClass.Car)
        )
        gt3 = ObjectTarget3D(
            [1, -1, 0], r, d,
            ObjectTag(KittiObjectClass.Pedestrian)
        )
        gt_boxes = ObjectTarget3DArray([gt1, gt2, gt3], frame="test")

        matcher.prepare_boxes(dt_boxes, gt_boxes, DistanceTypes.Euclidean)
        matcher.match(
            list(range(len(dt_boxes))), list(range(len(gt_boxes))),
            {KittiObjectClass.Car.value: 0.5, KittiObjectClass.Van.value: 0.5, KittiObjectClass.Pedestrian.value:0.7}
        )
        assert matcher.num_of_matches() == 0
