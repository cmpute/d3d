import numpy as np
from d3d.abstraction import ObjectTarget3DArray
from d3d.box import rbox_2d_iou

class ObjectBenchmark:
    def __init__(self, classes, min_overlaps, npr_samples=41):
        '''
        :param classes: Object classes to consider
        :param min_overlaps: Min overlaps per class for two boxes being considered as overlap.
        :param npr_samples: Number of precision-recall sample points
        '''
        self._classes = classes # TODO: convert enum to name
        self._min_overlaps = {classes[i]: v for i, v in enumerate(min_overlaps)}
        self._npr_samples = npr_samples

        self._total_gt = {k: 0 for k in self._classes}
        self._total_dt = {k: 0 for k in self._classes}
        self._tp = {k: 0 for k in self._classes}
        self._fp = {k: 0 for k in self._classes}
        self._fn = {k: 0 for k in self._classes}

    def _get_thresholds(self, scores):
        scores = np.sort(scores)[::-1]
        current_recall = 0
        thresholds = []
        for i, score in enumerate(scores):
            l_recall = (i + 1) / num_gt
            if i < (len(scores) - 1):
                r_recall = (i + 2) / num_gt
            else:
                r_recall = l_recall
            if (r_recall - current_recall) < (current_recall - l_recall) \
                and i < (len(scores) - 1):
                continue

            thresholds.append(score)
            current_recall += 1. / (self._npr_samples - 1)
        return thresholds

    def add_results(self, gt_boxes: ObjectTarget3DArray, dt_boxes: ObjectTarget3DArray):
        gt_array = gt_3dboxes.to_torch()
        dt_array = dt_boxes.to_torch()
        iou = rbox_2d_iou(gt_array[:, [0,1,3,4,6]], dt_array[:, [0,1,3,4,6]])

        dt_assignment = {}
        gt_assignment = {}
        thresholds = []
        for gt_idx in range(len(gt_boxes)):
            # skip classes not required
            gt_clsname = gt_boxes[gt_idx].tag_name
            if gt_clsname not in self._classes:
                return

            dt_matched = -1            
            for dt_idx in range(len(dt_boxes)):
                # skip already assigned box
                if dt_idx in dt_assignment:
                    continue
                # compare class information
                if dt_boxes[gt_idx].tag_name != gt_clsname:
                    continue

                if iou[gt_idx, dt_idx] > self._min_overlaps[gt_clsname]:
                    dt_assignment[dt_idx] = gt_idx
                    gt_assignment[gt_idx] = dt_idx # FIXME: store the best one?

            self._total_gt[gt_clsname] += 1
            if gt_idx in gt_assignment:
                self.tp += 1
                thresholds.append(det_boxes[dt_idx].tag_score)
            else:
                self.fn += 1

        # add statistics
        for dt_box in dt_boxes:
            self._total_dt[dt_box.tag_name] += 1

        self._tp += len(dt_assignment)
        self._


    def mAP(self):
        '''
        Calculate mean average precision (COCO)
        '''
        raise NotImplementedError()
