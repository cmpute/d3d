import unittest
import numpy as np

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import KittiObjectClass
from d3d.abstraction import Target3DArray, ObjectTarget3D, ObjectTag, TrackingTarget3D
from d3d.benchmarks import DetectionEvaluator, DetectionEvalStats, TrackingEvaluator

class TestDetectionEvaluator(unittest.TestCase):
    def test_calc_stats(self):
        eval_classes = [KittiObjectClass.Car, KittiObjectClass.Van]
        evaluator = DetectionEvaluator(eval_classes, [0.1, 0.2])

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
        dt_boxes = Target3DArray([dt1, dt2, dt3], frame="test")
        
        # test match with self
        result = evaluator.calc_stats(dt_boxes, dt_boxes)
        for clsobj in eval_classes:
            clsid = clsobj.value
            assert result.ngt[clsid] == 1
            assert result.ndt[clsid][0] == 1 and result.ndt[clsid][-1] == 0
            assert result.tp[clsid][0] == 1 and result.tp[clsid][-1] == 0
            assert result.fp[clsid][0] == 0 and result.fp[clsid][-1] == 0
            assert result.fn[clsid][0] == 0 and result.fn[clsid][-1] == 1
            assert np.isclose(result.acc_iou[clsid][0], 1) and np.isnan(result.acc_iou[clsid][-1])
            assert np.isclose(result.acc_angular[clsid][0], 0) and np.isnan(result.acc_angular[clsid][-1])
            assert np.isclose(result.acc_dist[clsid][0], 0) and np.isnan(result.acc_dist[clsid][-1])
            assert np.isclose(result.acc_box[clsid][0], 0) and np.isnan(result.acc_box[clsid][-1])
            assert np.isinf(result.acc_var[clsid][0]) and np.isnan(result.acc_var[clsid][-1])

        # test match with other boxes
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
        gt_boxes = Target3DArray([gt1, gt2, gt3], frame="test")
        result = evaluator.calc_stats(gt_boxes, dt_boxes)
        for clsobj in eval_classes:
            clsid = clsobj.value
            assert result.ngt[clsid] == 1
            assert result.ndt[clsid][0] == 1 and result.ndt[clsid][-1] == 0

            if clsobj == KittiObjectClass.Car:
                assert result.tp[clsid][0] == 1 and result.tp[clsid][-1] == 0
                assert result.fp[clsid][0] == 0 and result.fp[clsid][-1] == 0
                assert result.fn[clsid][0] == 0 and result.fn[clsid][-1] == 1
                assert result.acc_iou[clsid][0] > 0.1 and np.isnan(result.acc_iou[clsid][-1])
                assert result.acc_angular[clsid][0] > 0 and np.isnan(result.acc_angular[clsid][-1])
                assert result.acc_dist[clsid][0] > 1 and np.isnan(result.acc_dist[clsid][-1])
                assert result.acc_box[clsid][0] > 0 and np.isnan(result.acc_box[clsid][-1])
                assert np.isinf(result.acc_var[clsid][0]) and np.isnan(result.acc_var[clsid][-1])
            else:
                assert result.tp[clsid][0] == 0 and result.tp[clsid][-1] == 0
                assert result.fp[clsid][0] == 1 and result.fp[clsid][-1] == 0
                assert result.fn[clsid][0] == 1 and result.fn[clsid][-1] == 1
                assert np.isnan(result.acc_iou[clsid][0]) and np.isnan(result.acc_iou[clsid][-1])
                assert np.isnan(result.acc_angular[clsid][0]) and np.isnan(result.acc_angular[clsid][-1])
                assert np.isnan(result.acc_dist[clsid][0]) and np.isnan(result.acc_dist[clsid][-1])
                assert np.isnan(result.acc_box[clsid][0]) and np.isnan(result.acc_box[clsid][-1])
                assert np.isnan(result.acc_var[clsid][0]) and np.isnan(result.acc_var[clsid][-1])

    def test_pickling(self):
        '''
        Pickling functionality is essential for multiprocessing
        '''
        import pickle, io

        evaluator = DetectionEvaluator([KittiObjectClass.Car], [0.2])

        buffer = io.BytesIO()
        pickle.dump(evaluator, buffer)
        buffer.seek(0)
        evaluator_copy = pickle.load(buffer)

        assert np.allclose(evaluator.score_thresholds, evaluator_copy.score_thresholds)

        summary = DetectionEvalStats()
        summary.ngt = {1:1, 2:1}
        summary.ndt = {1:[2,2,1,1], 2:[2,1,1,1]}
        summary.acc_iou = {1:[0.2,0.2,0.1,0.2], 2:[0.2,0.1,0.1,0.1]}

        buffer = io.BytesIO()
        pickle.dump(summary, buffer)
        buffer.seek(0)
        summary_copy = pickle.load(buffer)
        assert summary.ngt == summary_copy.ngt
        assert summary.ndt == summary_copy.ndt
        assert summary.acc_iou == summary_copy.acc_iou


class TestTrackingEvaluator(unittest.TestCase):
    def test_scenarios(self):
        eval_classes = [KittiObjectClass.Car, KittiObjectClass.Van]
        evaluator = TrackingEvaluator(eval_classes, [0.5, 1])


        # scenario: X-crossing switch
        r = Rotation.from_euler("Z", 0)
        d = [1, 1, 1]
        t = ObjectTag(KittiObjectClass.Car, scores=0.8)
        v = [0, 0, 0]
        tid = 1
        traj1 = [
            TrackingTarget3D([-2, 2, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([-1, 1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 0, 0, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 1, 1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 2, 2, 0], r, d, v, v, t, tid=tid),
        ]
        t = ObjectTag(KittiObjectClass.Car, scores=0.9)
        tid = 2
        traj2 = [
            TrackingTarget3D([-2, -2, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([-1, -1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 0,  0, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 1, -1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 2, -2, 0], r, d, v, v, t, tid=tid),
        ]
        dt_trajs = [Target3DArray([t1, t2], frame="test") for t1, t2 in zip(traj1, traj2)]


        r = Rotation.from_euler("Z", 0.01)
        d = [1.1, 1.1, 1.1]
        t = ObjectTag(KittiObjectClass.Car)
        tid = 1001
        gt1 = [
            ObjectTarget3D([-2.1, 2.1, 0], r, d, t, tid=tid),
            ObjectTarget3D([-1.1, 0.9, 0], r, d, t, tid=tid),
            ObjectTarget3D([-0.1, 0.1, 0], r, d, t, tid=tid),
            ObjectTarget3D([0.9, -1.1, 0], r, d, t, tid=tid),
            ObjectTarget3D([1.9, -1.9, 0], r, d, t, tid=tid),
        ]
        tid = 1002
        gt2 = [
            ObjectTarget3D([-2.1, -2.1, 0], r, d, t, tid=tid),
            ObjectTarget3D([-1.1, -0.9, 0], r, d, t, tid=tid),
            ObjectTarget3D([-0.1,  0.1, 0], r, d, t, tid=tid),
            ObjectTarget3D([ 0.9,  1.1, 0], r, d, t, tid=tid),
            ObjectTarget3D([ 1.9,  1.9, 0], r, d, t, tid=tid),
        ]
        gt_trajs = [Target3DArray([t1, t2], frame="test") for t1, t2 in zip(gt1, gt2)]

        for dt_array, gt_array in zip(dt_trajs, gt_trajs):
            stats = evaluator.calc_stats(gt_array, dt_array)
            evaluator.add_stats(stats)
        
        assert evaluator.tp()[KittiObjectClass.Car] == 10
        assert evaluator.fp()[KittiObjectClass.Car] == 0
        assert evaluator.fn()[KittiObjectClass.Car] == 0
        assert evaluator.fn()[KittiObjectClass.Car] == 0
        assert evaluator.id_switches()[KittiObjectClass.Car] == 2
        assert evaluator.fragments()[KittiObjectClass.Car] == 2

        # scenario: X-crossing with 3 tracklets
        evaluator.reset()

        r = Rotation.from_euler("Z", 0)
        d = [1, 1, 1]
        t = ObjectTag(KittiObjectClass.Car, scores=0.8)
        v = [0, 0, 0]
        tid = 1
        traj1 = [
            TrackingTarget3D([-2, 2, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([-1, 1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 0, 0, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 1, 1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 2, 2, 0], r, d, v, v, t, tid=tid),
        ]
        t = ObjectTag(KittiObjectClass.Car, scores=0.9)
        tid = 2
        traj2 = [
            TrackingTarget3D([-2, -2, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([-1, -1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 0,  0, 0], r, d, v, v, t, tid=tid)
        ]
        tid = 3
        traj3 = [
            TrackingTarget3D([ 1, -1, 0], r, d, v, v, t, tid=tid),
            TrackingTarget3D([ 2, -2, 0], r, d, v, v, t, tid=tid)
        ]
        dt_trajs =  [Target3DArray([t2, t1], frame="test") for t1, t2 in zip(traj1[:3], traj2)]
        dt_trajs += [Target3DArray([t3, t1], frame="test") for t1, t3 in zip(traj1[3:], traj3)]

        for dt_array, gt_array in zip(dt_trajs, gt_trajs):
            stats = evaluator.calc_stats(gt_array, dt_array)
            evaluator.add_stats(stats)

        assert evaluator.tp()[KittiObjectClass.Car] == 10
        assert evaluator.fp()[KittiObjectClass.Car] == 0
        assert evaluator.fn()[KittiObjectClass.Car] == 0
        assert evaluator.fn()[KittiObjectClass.Car] == 0
        assert evaluator.id_switches()[KittiObjectClass.Car] == 2
        assert evaluator.fragments()[KittiObjectClass.Car] == 1
        assert evaluator.tracked_ratio()[KittiObjectClass.Car] == 1.
        assert evaluator.lost_ratio()[KittiObjectClass.Car] == 0.

if __name__ == "__main__":
    TestTrackingEvaluator().test_scenarios()
