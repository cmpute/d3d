import unittest
import numpy as np

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import KittiObjectClass
from d3d.abstraction import ObjectTarget3DArray, ObjectTarget3D, ObjectTag
from d3d.tracking.matcher import DistanceTypes, NearestNeighborMatcher, ScoreMatcher

class TestMatchers(unittest.TestCase):
    def set_up_matcher(self):
        # src boxes
        r = Rotation.from_euler("Z", 0)
        d = [2, 2, 2]
        dt1 = ObjectTarget3D(
            [0, 0, 0], r, d,
            ObjectTag(KittiObjectClass.Car, scores=0.8)
        )
        dt2 = ObjectTarget3D(
            [1, 1, 0], r, d,
            ObjectTag(KittiObjectClass.Van, scores=0.7)
        )
        dt3 = ObjectTarget3D(
            [-1, -1, 0], r, d,
            ObjectTag(KittiObjectClass.Car, scores=0.8)
        )
        dt_boxes = ObjectTarget3DArray([dt1, dt2, dt3], frame="test")

        # dst boxes
        r = Rotation.from_euler("Z", 0)
        d = [2, 2, 2]
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
            ObjectTag(KittiObjectClass.Van)
        )
        gt_boxes = ObjectTarget3DArray([gt1, gt2, gt3], frame="test")

        self.matcher_case1 = (dt_boxes, gt_boxes)

    def setUp(self):
        self.set_up_matcher()
        
    def test_nearest_neighbor_matcher(self):
        src_boxes, dst_boxes = self.matcher_case1

        matcher = NearestNeighborMatcher()
        matcher.prepare_boxes(src_boxes, dst_boxes, DistanceTypes.Position)
        matcher.match(
            list(range(len(src_boxes))), list(range(len(dst_boxes))),
            {KittiObjectClass.Car.value: 1.5, KittiObjectClass.Van.value: 1.5}
        )
        assert matcher.num_of_matches() == 2
        assert matcher.query_dst_match(1) in [0, 2]
        assert matcher.query_src_match(1) == 0

    def test_score_matcher(self):
        src_boxes, dst_boxes = self.matcher_case1

        matcher = ScoreMatcher()
        matcher.prepare_boxes(src_boxes, dst_boxes, DistanceTypes.Position)
        matcher.match(
            list(range(len(src_boxes))), list(range(len(dst_boxes))),
            {KittiObjectClass.Car.value: 1.5, KittiObjectClass.Van.value: 1.5}
        )
        assert matcher.num_of_matches() == 2
        assert matcher.query_dst_match(1) in [0, 2]
        assert matcher.query_src_match(1) == 0
