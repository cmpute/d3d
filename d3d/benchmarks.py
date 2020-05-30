import math

import numpy as np
import torch
from addict import Dict as edict

from d3d.abstraction import ObjectTarget3DArray
from d3d.box import box2d_iou


class ObjectBenchmark:
    def __init__(self, classes, min_overlaps, pr_sample_count=40, min_score=0, pr_sample_scale="log10"):
        '''
        Object detection benchmark. Targets association is done by score sorting.

        :param classes: Object classes to consider
        :param min_overlaps: Min overlaps per class for two boxes being considered as overlap.
            If single value is provided, all class will use the same overlap threshold
        :param min_score: Min score for precision-recall samples
        :param pr_sample_count: Number of precision-recall sample points (expect for p=1,r=0 and p=0,r=1)
        :param pr_sample_scale: PR sample type, {lin: linspace, log: logspace 1~10, logX: logspace 1~X}

        TODO: add support for other threshold (e.g. center distance)
        '''
        # parse parameters
        if isinstance(classes, (list, tuple)):
            self._classes = classes
        else:
            self._classes = [classes]
        if isinstance(min_overlaps, (list, tuple)):
            self._min_overlaps = {classes[i]: v for i, v in enumerate(min_overlaps)}
        else:
            self._min_overlaps = {c: min_overlaps for c in self._classes}

        self._pr_nsamples = pr_sample_count
        self._min_score = min_score

        # generate score thresholds
        if pr_sample_scale == "lin":
            self._pr_thresholds = np.linspace(min_score, 1, pr_sample_count, endpoint=False)
        elif pr_sample_scale.startswith("log"):
            logstart, logend = 1, int(pr_sample_scale[3:] or "10")
            self._pr_thresholds = np.geomspace(logstart, logend, pr_sample_count+1)
            self._pr_thresholds = (self._pr_thresholds - logstart) * (1 - min_score) / (logend - logstart)
            self._pr_thresholds = (1 - self._pr_thresholds)[:1:-1]
        else:
            raise ValueError("Unrecognized PR sample type")

        self.reset()

    def reset(self):
        # aggregated statistics
        self._total_gt = {k: 0 for k in self._classes}
        self._total_dt = {k: [0] * self._pr_nsamples for k in self._classes}
        self._tp = {k: [0] * self._pr_nsamples for k in self._classes}
        self._fp = {k: [0] * self._pr_nsamples for k in self._classes}
        self._fn = {k: [0] * self._pr_nsamples for k in self._classes}

        self._acc_angular = {k: [0] * self._pr_nsamples for k in self._classes}
        self._acc_iou = {k: [0] * self._pr_nsamples for k in self._classes}
        self._acc_box = {k: [0] * self._pr_nsamples for k in self._classes}
        self._acc_dist = {k: [0] * self._pr_nsamples for k in self._classes}

    def get_stats(self, gt_boxes: ObjectTarget3DArray, dt_boxes: ObjectTarget3DArray):
        assert gt_boxes.frame == dt_boxes.frame        

        # statistics
        tp = {k: [0] * self._pr_nsamples for k in self._classes}
        fp = {k: [0] * self._pr_nsamples for k in self._classes}
        fn = {k: [0] * self._pr_nsamples for k in self._classes}
        ngt = {k: 0 for k in self._classes}
        ndt = {k: [0] * self._pr_nsamples for k in self._classes}

        dt_assignment = [{} for _ in range(self._pr_nsamples)]
        gt_assignment = [{} for _ in range(self._pr_nsamples)]

        iou_acc = [{} for _ in range(self._pr_nsamples)]
        angular_acc = [{} for _ in range(self._pr_nsamples)]
        dist_acc = [{} for _ in range(self._pr_nsamples)]
        box_acc = [{} for _ in range(self._pr_nsamples)]

        # calculate iou and sort by score
        gt_array = gt_boxes.to_torch().float() # TODO: implement type dispatching
        dt_array = dt_boxes.to_torch().float()
        iou = box2d_iou(gt_array[:, [0,1,3,4,6]], dt_array[:, [0,1,3,4,6]], method="rbox") # TODO: method "box" has negative values...
        order = np.argsort([box.tag_score for box in dt_boxes])[::-1] # match from best score

        for gt_idx in range(len(gt_boxes)):
            # skip classes not required
            gt_tag = gt_boxes[gt_idx].tag_top
            if gt_tag not in self._classes:
                continue
           
            for dt_idx in order:
                # compare class information
                if dt_boxes[dt_idx].tag_top != gt_tag:
                    continue

                # true positive if overlap is larger than threshold
                if iou[gt_idx, dt_idx] > self._min_overlaps[gt_tag]:
                    thres_loc = np.searchsorted(self._pr_thresholds, dt_boxes[dt_idx].tag_score)
                    assert thres_loc >= 0, "Box score should be larger than min_score!"

                    # assign box
                    for score_idx in range(0, thres_loc):
                        # skip already assigned box
                        if dt_idx in dt_assignment[score_idx]:
                            continue

                        dt_assignment[score_idx][dt_idx] = gt_idx
                        gt_assignment[score_idx][gt_idx] = dt_idx

                        # caculate accuracy values for various criteria
                        iou_acc[score_idx][gt_idx] = iou[gt_idx, dt_idx]
                        dist_acc[score_idx][gt_idx] = np.linalg.norm(
                            gt_boxes[gt_idx].position - dt_boxes[dt_idx].position)
                        box_acc[score_idx][gt_idx] = np.linalg.norm(
                            gt_boxes[gt_idx].dimension - dt_boxes[dt_idx].dimension)

                        angular_acc_cur = abs(gt_boxes[gt_idx].yaw - dt_boxes[dt_idx].yaw)
                        angular_acc_cur = min(angular_acc_cur, 2*math.pi - angular_acc_cur) / math.pi
                        angular_acc[score_idx][gt_idx] = angular_acc_cur
                    break

            ngt[gt_tag] += 1
            for score_idx in range(self._pr_nsamples):
                if gt_idx in gt_assignment[score_idx]:
                    tp[gt_tag][score_idx] += 1
                else:
                    fn[gt_tag][score_idx] += 1

        # compute false positives
        for dt_idx, dt_box in enumerate(dt_boxes):
            if dt_box.tag_top not in self._classes:
                continue

            thres_loc = np.searchsorted(self._pr_thresholds, dt_boxes[dt_idx].tag_score) - 1
            for score_idx in range(thres_loc):
                ndt[dt_box.tag_top][score_idx] += 1
                if dt_idx not in dt_assignment[score_idx]:
                    fp[dt_box.tag_top][score_idx] += 1

        # compute accuracy metrics
        def aggregate(acc):
            aggregated = {k: [[] for _ in range(self._pr_nsamples)] for k in self._classes}
            for score_idx, score_acc in enumerate(acc):
                for gt_idx, gt_acc in score_acc.items():
                    aggregated[gt_boxes[gt_idx].tag_top][score_idx].append(gt_acc)
            for k in self._classes:
                for score_idx in range(self._pr_nsamples):
                    # assert len(aggregated[k][score_idx]) == tp[k][score_idx]
                    if len(aggregated[k][score_idx]) > 0:
                        aggregated[k][score_idx] = np.mean(aggregated[k][score_idx])
                    else:
                        aggregated[k][score_idx] = np.nan
            return aggregated

        return edict(ngt=ngt, ndt=ndt,
            tp=tp, fp=fp, fn=fn,
            acc_iou=aggregate(iou_acc),
            acc_angular=aggregate(angular_acc),
            acc_dist=aggregate(dist_acc),
            acc_box=aggregate(box_acc)
        )

    def add_stats(self, stats):
        '''
        Add statistics from get_stats into database
        '''
        def weighted_mean(a, wa, b, wb):
            if wa == 0 and wb == 0: return 0
            elif wa == 0: return b
            elif wb == 0: return a
            else: return (a * wa + b * wb) / (wa + wb)

        for k in self._classes:
            self._total_gt[k] += stats.ngt[k]
            for i in range(self._pr_nsamples):
                # aggregate accuracies
                otp, ntp = self._tp[k][i], stats.tp[k][i]
                self._acc_angular[k][i] = weighted_mean(
                    self._acc_angular[k][i], otp, stats.acc_angular[k][i], ntp)
                self._acc_box[k][i] = weighted_mean(
                    self._acc_box[k][i], otp, stats.acc_box[k][i], ntp)
                self._acc_iou[k][i] = weighted_mean(
                    self._acc_iou[k][i], otp, stats.acc_iou[k][i], ntp)
                self._acc_dist[k][i] = weighted_mean(
                    self._acc_dist[k][i], otp, stats.acc_dist[k][i], ntp)

                # aggregate common stats
                self._total_dt[k][i] += stats.ndt[k][i]
                self._tp[k][i] += stats.tp[k][i]
                self._fp[k][i] += stats.fp[k][i]
                self._fn[k][i] += stats.fn[k][i]


    def _get_score_idx(self, score=None):
        if score is None:
            return self._pr_nsamples // 2
        else:
            return np.searchsorted(self._pr_thresholds, score) - 1
    def gt_count(self):
        return self._total_gt
    def dt_count(self, score=None):
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._total_dt.items()}

    def tp(self, score=None):
        '''Return true positive count. If score is not specified, return the median value'''
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._tp.items()}
    def fp(self, score=None):
        '''Return false positive count. If score is not specified, return the median value'''
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._fp.items()}
    def fn(self, score=None):
        '''Return false negative count. If score is not specified, return the median value'''
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._fn.items()}

    def precision(self, score=None):
        if score is None:
            p = {k: [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    if self._fp[k][i] == 0:
                        p[k][i] = 1
                    else:
                        p[k][i] = self._tp[k][i] / (self._tp[k][i] + self._fp[k][i])
        else:
            tp, fp = self.tp(score), self.fp(score)
            p = {k: tp[k] / (tp[k]+fp[k]) if fp[k] > 0 else 1 for k in self._classes}
        return p
    def recall(self, score=None):
        if score is None:
            r = {k: [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    if self._fn[k][i] == 0:
                        r[k][i] = 1
                    else:
                        r[k][i] = self._tp[k][i] / (self._tp[k][i] + self._fn[k][i])
        else:
            tp, fn = self.tp(score), self.fn(score)
            r = {k: tp[k] / (tp[k]+fn[k]) if fn[k] > 0 else 1 for k in self._classes}
        return r

    def fscore(self, beta=1, score=None):
        b2 = beta * beta
        if score is None:
            p, r = self.precision(), self.recall()
            fs = {k: [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    fs[k][i] = (1+b2) * (p[k][i] + r[k][i]) / (b2*p[k][i] + r[k][i])
        else:
            p, r = self.precision(score), self.recall(score)
            fs = {k: (1+b2) * (p[k]+r[k]) / (b2*p[k]+r[k]) for k in self._classes}
        return r

    def ap(self):
        '''Calculate (mean) average precision'''
        p, r = self.precision(), self.recall()
        # usually pr curve grows from bottom right to top left as score threshold
        # increases, so the area can be negative
        area = {k: -np.trapz(p[k], r[k]) for k in self._classes}
        return area

    def acc_iou(self, score=None):
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._acc_iou.items()}
    def acc_box(self, score=None):
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._acc_box.items()}
    def acc_dist(self, score=None):
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._acc_dist.items()}
    def acc_angular(self, score=None):
        score_idx = self._get_score_idx(score)
        return {k: v[score_idx] for k, v in self._acc_angular.items()}

    def summary(self):
        '''
        Print default summary (into returned string)
        '''
        score_thres = 0.8

        lines = [''] # prepend an empty line
        lines.append("========== Benchmark Summary ==========")
        precision, recall = self.precision(score_thres), self.recall(score_thres)
        fscore, ap = self.fscore(score_thres), self.ap()
        acc_iou, acc_angular = self.acc_iou(score_thres), self.acc_angular(score_thres)
        acc_dist, acc_box = self.acc_dist(score_thres), self.acc_box(score_thres)

        for k in self._classes:
            lines.append("Results for %s:" % k.name)
            lines.append("\tTotal processed targets:\t%d gt boxes, %d dt boxes" % (
                self._total_gt[k], max(self._total_dt[k])
            ))
            lines.append("\tPrecision (score > %.2f):\t%.3f" % (score_thres, precision[k]))
            lines.append("\tRecall (score > %.2f):\t\t%.3f" % (score_thres, recall[k]))
            lines.append("\tMax F1:\t\t\t\t%.3f" % max(fscore[k]))
            lines.append("\tAP:\t\t\t\t%.3f" % ap[k])
            lines.append("")
            lines.append("\tMean IoU (score > %.2f):\t\t%.3f" % (score_thres, acc_iou[k]))
            lines.append("\tMean angular error (score > %.2f):\t%.3f" % (score_thres, acc_angular[k]))
            lines.append("\tMean distance (score > %.2f):\t\t%.3f" % (score_thres, acc_dist[k]))
            lines.append("\tMean box error (score > %.2f):\t\t%.3f" % (score_thres, acc_box[k]))
        lines.append("========== Summary End ==========")

        return '\n'.join(lines)
