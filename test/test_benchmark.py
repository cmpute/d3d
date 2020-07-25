import unittest
import numpy as np

from scipy.spatial.transform import Rotation
from d3d.dataset.kitti import KittiObjectClass
from d3d.abstraction import ObjectTarget3DArray, ObjectTarget3D, ObjectTag
from d3d.benchmarks import DetectionEvaluator, DetectionEvalStats

class TestBenchmark(unittest.TestCase):
    def test_get_stats(self):
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
        dt_boxes = ObjectTarget3DArray([dt1, dt2, dt3], frame="test")
        
        # test match with self
        result = evaluator.get_stats(dt_boxes, dt_boxes)
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
        gt_boxes = ObjectTarget3DArray([gt1, gt2, gt3], frame="test")
        result = evaluator.get_stats(gt_boxes, dt_boxes)
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
