# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cimport cython
import numpy as np
cimport numpy as np
import scipy.stats as sps
import torch
from addict import Dict as edict

from numpy.math cimport NAN, isnan, PI, isinf
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

from d3d.abstraction cimport ObjectTarget3DArray
from d3d.tracking.matcher cimport ScoreMatcher, DistanceTypes
from d3d.box import box2d_iou
from d3d.math cimport wmean, diffnorm3

cdef inline int bisect(vector[float] &arr, float x) nogil:
    '''Cython version of bisect.bisect_left'''
    cdef int lo=0, hi=arr.size(), mid
    while lo < hi:
        mid = (lo+hi)//2
        if arr[mid] < x: lo = mid+1
        else: hi = mid
    return lo

cdef inline float calc_precision(int tp, int fp) nogil:
    if fp == 0: return 1
    else: return <float>tp / (tp + fp)
cdef inline float calc_recall(int tp, int fn) nogil:
    if fn == 0: return 1
    else: return <float>tp / (tp + fn)
cdef inline float calc_fscore(int tp, int fp, int fn, float b2) nogil:
    return (1+b2) * tp / ((1+b2)*tp + b2*fn + fp)

@cython.auto_pickle(True)
cdef class DetectionEvalStats:
    '''Stats summary of a evaluation step'''
    cdef public unordered_map[int, vector[float]] acc_iou, acc_angular, acc_dist, acc_box, acc_var
    cdef public unordered_map[int, int] ngt
    cdef public unordered_map[int, vector[int]] tp, fp, fn, ndt

@cython.auto_pickle(True)
cdef class DetectionEvaluator:
    '''Benchmark for object detection'''

    # member declarations
    cdef int _pr_nsamples
    cdef float _min_score
    cdef unordered_set[int] _classes
    cdef object _class_type
    cdef unordered_map[int, float] _min_overlaps
    cdef vector[float] _pr_thresholds

    # aggregated statistics declarations
    cdef unordered_map[int, int] _total_gt
    cdef unordered_map[int, vector[int]] _total_dt, _tp, _fp, _fn
    cdef unordered_map[int, vector[float]] _acc_angular, _acc_iou, _acc_box, _acc_dist, _acc_var

    def __init__(self, classes, min_overlaps, int pr_sample_count=40, float min_score=0, str pr_sample_scale="log10"):
        '''
        Object detection benchmark. Targets association is done by score sorting.

        :param classes: Object classes to consider
        :param min_overlaps: Min overlaps per class for two boxes being considered as overlap.
            If single value is provided, all class will use the same overlap threshold
        :param min_score: Min score for precision-recall samples
        :param pr_sample_count: Number of precision-recall sample points (expect for p=1,r=0 and p=0,r=1)
        :param pr_sample_scale: PR sample type, {lin: linspace, log: logspace 1~10, logX: logspace 1~X}
        '''
        # parse parameters
        if isinstance(classes, (list, tuple)):
            assert len(classes) > 0
            self._class_type = type(classes[0])
            for c in classes:
                self._classes.insert(c.value)
        else:
            self._class_type = type(classes)
            self._classes.insert(classes.value)
        if isinstance(min_overlaps, (list, tuple)):
            # take negative since iou distance is defined as negative
            self._min_overlaps = {classes[i].value: -v for i, v in enumerate(min_overlaps)}
        elif isinstance(min_overlaps, (int, float)):
            self._min_overlaps = {c: -min_overlaps for c in self._classes}
        else:
            raise ValueError("min_overlaps should be a list or a single value")

        self._pr_nsamples = pr_sample_count
        self._min_score = min_score

        # generate score thresholds
        cdef np.ndarray thresholds
        if pr_sample_scale == "lin":
            thresholds = np.linspace(min_score, 1, pr_sample_count, endpoint=False, dtype=np.float32)
        elif pr_sample_scale.startswith("log"):
            logstart, logend = 1, int(pr_sample_scale[3:] or "10")
            thresholds = np.geomspace(logstart, logend, pr_sample_count+1, dtype=np.float32)
            thresholds = (thresholds - logstart) * (1 - min_score) / (logend - logstart)
            thresholds = (1 - thresholds)[:0:-1]
        else:
            raise ValueError("Unrecognized PR sample type")
        self._pr_thresholds = thresholds

        # initialize maps
        for k in self._classes:
            self._total_gt[k] = 0
            self._total_dt[k] = vector[int](self._pr_nsamples, 0)
            self._tp[k] = vector[int](self._pr_nsamples, 0)
            self._fp[k] = vector[int](self._pr_nsamples, 0)
            self._fn[k] = vector[int](self._pr_nsamples, 0)

            self._acc_angular[k] = vector[float](self._pr_nsamples, NAN)
            self._acc_iou[k] = vector[float](self._pr_nsamples, NAN)
            self._acc_box[k] = vector[float](self._pr_nsamples, NAN)
            self._acc_dist[k] = vector[float](self._pr_nsamples, NAN)
            self._acc_var[k] = vector[float](self._pr_nsamples, NAN)

    def reset(self):
        for k in self._classes:
            self._total_gt[k] = 0
            self._total_dt[k].assign(self._pr_nsamples, 0)
            self._tp[k].assign(self._pr_nsamples, 0)
            self._fp[k].assign(self._pr_nsamples, 0)
            self._fn[k].assign(self._pr_nsamples, 0)

            self._acc_angular[k].assign(self._pr_nsamples, NAN)
            self._acc_iou[k].assign(self._pr_nsamples, NAN)
            self._acc_box[k].assign(self._pr_nsamples, NAN)
            self._acc_dist[k].assign(self._pr_nsamples, NAN)
            self._acc_var[k].assign(self._pr_nsamples, NAN)

    cdef inline unordered_map[int, vector[float]] _aggregate_stats(self,
        vector[unordered_map[int, float]]& acc, vector[int]& gt_tags) nogil:
        '''Help put accuracy values into categories'''
        # init intermediate vars
        cdef unordered_map[int, vector[float]] sorted_sum, aggregated
        cdef unordered_map[int, vector[int]] sorted_count
        for k in self._classes:
            sorted_sum[k] = vector[float](self._pr_nsamples, 0)
            sorted_count[k] = vector[int](self._pr_nsamples, 0)
            aggregated[k] = vector[float](self._pr_nsamples, 0)

        # sort accuracies into categories
        for score_idx in range(self._pr_nsamples):
            for iter in acc[score_idx]:
                sorted_sum[gt_tags[iter.first]][score_idx] += iter.second
                sorted_count[gt_tags[iter.first]][score_idx] += 1

        # aggregate
        for k in self._classes:
            for score_idx in range(self._pr_nsamples):
                # assert sorted_count[k][score_idx] == tp[k][score_idx]
                if sorted_count[k][score_idx] > 0:
                    aggregated[k][score_idx] = sorted_sum[k][score_idx] / sorted_count[k][score_idx]
                else:
                    aggregated[k][score_idx] = NAN
        return aggregated

    cpdef DetectionEvalStats get_stats(self, ObjectTarget3DArray gt_boxes, ObjectTarget3DArray dt_boxes):
        assert gt_boxes.frame == dt_boxes.frame        

        # forward definitions
        cdef int thres_loc, gt_tag, dt_idx, dt_tag

        # initialize matcher
        cdef ScoreMatcher matcher = ScoreMatcher()
        matcher.prepare_boxes(dt_boxes, gt_boxes, DistanceTypes.RIoU)

        # initialize statistics
        cdef DetectionEvalStats summary = DetectionEvalStats()
        cdef vector[unordered_map[int, float]] iou_acc, angular_acc, dist_acc, box_acc, var_acc
        for k in self._classes:
            summary.ngt[k] = 0
            summary.ndt[k] = vector[int](self._pr_nsamples, 0)
            summary.tp[k] = vector[int](self._pr_nsamples, 0)
            summary.fp[k] = vector[int](self._pr_nsamples, 0)
            summary.fn[k] = vector[int](self._pr_nsamples, 0)

            iou_acc.resize(self._pr_nsamples)
            angular_acc.resize(self._pr_nsamples)
            dist_acc.resize(self._pr_nsamples)
            box_acc.resize(self._pr_nsamples)
            var_acc.resize(self._pr_nsamples)

        
        # select ground-truth boxes to match
        cdef vector[int] gt_indices, dt_indices
        for gt_idx in range(len(gt_boxes)):
            gt_tag = gt_boxes.get(gt_idx).tag.labels[0]
            if self._min_overlaps.find(gt_tag) == self._min_overlaps.end():
                continue

            gt_indices.push_back(gt_idx)
            summary.ngt[gt_tag] += 1

        # loop over score thres
        for score_idx in range(self._pr_nsamples):
            score_thres = self._pr_thresholds[score_idx]

            # select detection boxes to match
            dt_indices.clear()
            for dt_idx in range(len(dt_boxes)):
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]
                if self._min_overlaps.find(dt_tag) == self._min_overlaps.end():
                    continue
                if dt_boxes[dt_idx].tag_score < score_thres:
                    continue

                dt_indices.push_back(dt_idx)
                summary.ndt[dt_tag][score_idx] += 1

            # match boxes
            matcher.match(dt_indices, gt_indices, self._min_overlaps)

            # process ground-truth match results
            for gt_idx in gt_indices:
                gt_tag = gt_boxes.get(gt_idx).tag.labels[0]
                dt_idx = matcher.query_dst_match(gt_idx)
                if dt_idx < 0:
                    summary.fn[gt_tag][score_idx] += 1
                    continue
                summary.tp[gt_tag][score_idx] += 1

                # caculate accuracy values for various criteria
                iou_acc[score_idx][gt_idx] = -matcher._distance_cache[dt_idx, gt_idx] # FIXME: not elegant here
                dist_acc[score_idx][gt_idx] = diffnorm3(gt_boxes.get(gt_idx).position_, dt_boxes.get(dt_idx).position_)
                box_acc[score_idx][gt_idx] = diffnorm3(gt_boxes.get(gt_idx).dimension_, dt_boxes.get(dt_idx).dimension_)

                angular_acc_cur = (gt_boxes[gt_idx].orientation.inv() * dt_boxes[dt_idx].orientation).magnitude()
                angular_acc[score_idx][gt_idx] = angular_acc_cur / PI

                if dt_boxes[dt_idx].orientation_var > 0:
                    var_acc_cur = sps.multivariate_normal.logpdf(gt_boxes[gt_idx].position,
                        dt_boxes[dt_idx].position, cov=dt_boxes[dt_idx].position_var)
                    var_acc_cur += sps.multivariate_normal.logpdf(gt_boxes[gt_idx].dimension,
                        dt_boxes[dt_idx].dimension, cov=dt_boxes[dt_idx].dimension_var)
                    var_acc_cur += sps.vonmises.logpdf(angular_acc_cur, kappa=1/dt_boxes[dt_idx].orientation_var)
                    var_acc[score_idx][gt_idx] = var_acc_cur
                else:
                    var_acc[score_idx][gt_idx] = -np.inf

            # process detection match results
            for dt_idx in dt_indices:
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]     
                if matcher.query_src_match(dt_idx) < 0:
                    summary.fp[dt_tag][score_idx] += 1

        # compute accuracy metrics
        cdef vector[int] gt_tags
        gt_tags.reserve(len(gt_boxes))
        for gt_idx in range(len(gt_boxes)):
            gt_tags.push_back(gt_boxes.get(gt_idx).tag.labels[0])

        summary.acc_iou = self._aggregate_stats(iou_acc, gt_tags)
        summary.acc_angular = self._aggregate_stats(angular_acc, gt_tags)
        summary.acc_dist = self._aggregate_stats(dist_acc, gt_tags)
        summary.acc_box = self._aggregate_stats(box_acc, gt_tags)
        summary.acc_var = self._aggregate_stats(var_acc, gt_tags)
        return summary

    cpdef add_stats(self, DetectionEvalStats stats):
        '''
        Add statistics from get_stats into database
        '''
        cdef int otp, ntp
        for k in self._classes:
            self._total_gt[k] += stats.ngt[k]
            for i in range(self._pr_nsamples):
                # aggregate accuracies
                otp, ntp = self._tp[k][i], stats.tp[k][i]
                self._acc_angular[k][i] = wmean(
                    self._acc_angular[k][i], otp, stats.acc_angular[k][i], ntp)
                self._acc_box[k][i] = wmean(
                    self._acc_box[k][i], otp, stats.acc_box[k][i], ntp)
                self._acc_iou[k][i] = wmean(
                    self._acc_iou[k][i], otp, stats.acc_iou[k][i], ntp)
                self._acc_dist[k][i] = wmean(
                    self._acc_dist[k][i], otp, stats.acc_dist[k][i], ntp)
                self._acc_var[k][i] = wmean(
                    self._acc_var[k][i], otp, stats.acc_var[k][i], ntp)

                # aggregate common stats
                self._total_dt[k][i] += stats.ndt[k][i]
                self._tp[k][i] += stats.tp[k][i]
                self._fp[k][i] += stats.fp[k][i]
                self._fn[k][i] += stats.fn[k][i]

    cdef inline int _get_score_idx(self, float score) nogil:
        if isnan(score):
            return self._pr_nsamples // 2
        else:
            return bisect(self._pr_thresholds, score)

    @property
    def score_thresholds(self):
        return np.asarray(self._pr_thresholds)

    def gt_count(self):
        return self._total_gt
    def dt_count(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._total_dt}

    def tp(self, float score=NAN):
        '''Return true positive count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._tp}
    def fp(self, float score=NAN):
        '''Return false positive count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._fp}
    def fn(self, float score=NAN):
        '''Return false negative count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._fn}

    def precision(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        if isnan(score):
            p = {k: [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    p[k][i] = calc_precision(self._tp[k][i], self._fp[k][i])
        else:
            p = {k: calc_precision(self._tp[k][score_idx], self._fp[k][score_idx]) for k in self._classes}
        return p
    def recall(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        if isnan(score):
            r = {k: [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    r[k][i] = calc_recall(self._tp[k][i], self._fn[k][i])
        else:
            r = {k: calc_recall(self._tp[k][score_idx], self._fn[k][score_idx]) for k in self._classes}
        return r
    def fscore(self, float score=NAN, float beta=1):
        cdef float b2 = beta * beta        
        cdef int score_idx = self._get_score_idx(score)
        if isnan(score):
            fs = {k: [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    fs[k][i] = calc_fscore(self._tp[k][i], self._fp[k][i], self._fn[k][i], b2)
        else:
            fs = {k: calc_fscore(self._tp[k][score_idx], self._fp[k][score_idx], self._fn[k][score_idx], b2)
                for k in self._classes}
        return fs

    def ap(self):
        '''Calculate (mean) average precision'''
        p, r = self.precision(), self.recall()
        # usually pr curve grows from bottom right to top left as score threshold
        # increases, so the area can be negative
        area = {k: -np.trapz(p[k], r[k]) for k in self._classes}
        return area

    def acc_iou(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._acc_iou}
    def acc_box(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._acc_box}
    def acc_dist(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._acc_dist}
    def acc_angular(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(iter.first): iter.second[score_idx] for iter in self._acc_angular}

    def summary(self, float score_thres = 0.8):
        '''
        Print default summary (into returned string)
        '''
        cdef int score_idx = self._get_score_idx(score_thres)

        cdef list lines = [''] # prepend an empty line
        precision, recall = self.precision(score_thres), self.recall(score_thres)
        fscore, ap = self.fscore(), self.ap()

        lines.append("========== Benchmark Summary ==========")
        for k in self._classes:
            lines.append("Results for %s:" % self._class_type(k).name)
            lines.append("\tTotal processed targets:\t%d gt boxes, %d dt boxes" % (
                self._total_gt[k], max(self._total_dt[k])
            ))
            lines.append("\tPrecision (score > %.2f):\t%.3f" % (score_thres, precision[k]))
            lines.append("\tRecall (score > %.2f):\t\t%.3f" % (score_thres, recall[k]))
            lines.append("\tMax F1:\t\t\t\t%.3f" % max(fscore[k]))
            lines.append("\tAP:\t\t\t\t%.3f" % ap[k])
            lines.append("")
            lines.append("\tMean IoU (score > %.2f):\t\t%.3f" % (score_thres, self._acc_iou[k][score_idx]))
            lines.append("\tMean angular error (score > %.2f):\t%.3f" % (score_thres, self._acc_angular[k][score_idx]))
            lines.append("\tMean distance (score > %.2f):\t\t%.3f" % (score_thres, self._acc_dist[k][score_idx]))
            lines.append("\tMean box error (score > %.2f):\t\t%.3f" % (score_thres, self._acc_box[k][score_idx]))
            if not isinf(self._acc_var[k][score_idx]):
                lines.append("\tMean variance error (score > %.2f):\t%.3f" % (score_thres, self._acc_var[k][score_idx]))
        lines.append("========== Summary End ==========")

        return '\n'.join(lines)
