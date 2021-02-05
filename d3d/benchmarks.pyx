# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, embedsignature=True

cimport cython
import numpy as np
cimport numpy as np
import scipy.stats as sps
import torch
from addict import Dict as edict

from numpy.math cimport NAN, isnan, PI, isinf, INFINITY
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

from d3d.abstraction cimport Target3DArray, TransformSet
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
    ''' Detection stats summary of a evaluation step '''
    cdef public unordered_map[int, int] ngt
    cdef public unordered_map[int, vector[int]] tp, fp, fn, ndt
    cdef public unordered_map[int, vector[float]] acc_iou, acc_angular, acc_dist, acc_box, acc_var

    cdef void initialize(self, unordered_set[int] classes, int nsamples):
        for k in classes:
            self.ngt[k] = 0
            self.ndt[k] = vector[int](nsamples, 0)
            self.tp[k] = vector[int](nsamples, 0)
            self.fp[k] = vector[int](nsamples, 0)
            self.fn[k] = vector[int](nsamples, 0)

            self.acc_angular[k] = vector[float](nsamples, NAN)
            self.acc_iou[k] = vector[float](nsamples, NAN)
            self.acc_box[k] = vector[float](nsamples, NAN)
            self.acc_dist[k] = vector[float](nsamples, NAN)
            self.acc_var[k] = vector[float](nsamples, NAN)

    def as_object(self):
        return dict(ngt=self.ngt, tp=self.tp, fp=self.fp, fn=self.fn, ndt=self.ndt,
            acc_iou=self.acc_iou, acc_angular=self.acc_angular, acc_dist=self.acc_dist,
            acc_box=self.acc_box, acc_var=self.acc_var
        )

@cython.auto_pickle(True)
cdef class DetectionEvaluator:
    '''Benchmark for object detection'''

    # member declarations
    cdef int _pr_nsamples
    cdef float _min_score
    cdef unordered_set[int] _classes
    cdef object _class_type
    cdef unordered_map[int, float] _max_distance
    cdef vector[float] _pr_thresholds
    cdef DetectionEvalStats _stats # aggregated statistics declarations

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
            self._max_distance = {classes[i].value: 1 - v for i, v in enumerate(min_overlaps)}
        elif isinstance(min_overlaps, (int, float)):
            self._max_distance = {c: 1 - min_overlaps for c in self._classes}
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
        self._stats = DetectionEvalStats()
        self._stats.initialize(self._classes, self._pr_nsamples)

    cpdef void reset(self):
        self._stats.initialize(self._classes, self._pr_nsamples)

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
            for diter in acc[score_idx]:
                sorted_sum[gt_tags[diter.first]][score_idx] += diter.second
                sorted_count[gt_tags[diter.first]][score_idx] += 1

        # aggregate
        for k in self._classes:
            for score_idx in range(self._pr_nsamples):
                # assert sorted_count[k][score_idx] == tp[k][score_idx]
                if sorted_count[k][score_idx] > 0:
                    aggregated[k][score_idx] = sorted_sum[k][score_idx] / sorted_count[k][score_idx]
                else:
                    aggregated[k][score_idx] = NAN
        return aggregated

    cpdef DetectionEvalStats calc_stats(self, Target3DArray gt_boxes, Target3DArray dt_boxes, TransformSet calib=None):
        # convert boxes to the same frame
        if gt_boxes.frame != dt_boxes.frame:
            if calib is None:
                raise ValueError("Calibration is not provided when dt_boxes and gt_boxes are in different frames!")
            gt_boxes = calib.transform_objects(gt_boxes, frame_to=dt_boxes.frame)

        # forward definitions
        cdef int gt_idx, gt_tag, dt_idx, dt_tag
        cdef float score_thres, angular_acc_cur, var_acc_cur

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
        for gt_idx in range(gt_boxes.size()):
            gt_tag = gt_boxes.get(gt_idx).tag.labels[0]
            if self._classes.find(gt_tag) == self._classes.end():
                continue  # skip objects within ignored category

            summary.ngt[gt_tag] += 1
            gt_indices.push_back(gt_idx)

        # loop over score thres
        for score_idx in range(self._pr_nsamples):
            score_thres = self._pr_thresholds[score_idx]

            # select detection boxes to match
            dt_indices.clear()
            for dt_idx in range(dt_boxes.size()):
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]
                if self._classes.find(dt_tag) == self._classes.end():
                    continue  # skip objects within ignored category
                if dt_boxes.get(dt_idx).tag.scores[0] < score_thres:
                    continue  # skip objects with lower scores

                summary.ndt[dt_tag][score_idx] += 1
                dt_indices.push_back(dt_idx)

            # match boxes
            matcher.clear_match()
            matcher.match(dt_indices, gt_indices, self._max_distance)

            # process ground-truth match results
            for gt_idx in gt_indices:
                gt_tag = gt_boxes.get(gt_idx).tag.labels[0]
                dt_idx = matcher.query_dst_match(gt_idx)
                if dt_idx < 0:
                    summary.fn[gt_tag][score_idx] += 1
                    continue
                summary.tp[gt_tag][score_idx] += 1

                # caculate accuracy values for various criteria
                iou_acc[score_idx][gt_idx] = 1 - matcher._distance_cache[dt_idx, gt_idx] # FIXME: not elegant here
                dist_acc[score_idx][gt_idx] = diffnorm3(gt_boxes.get(gt_idx).position_, dt_boxes.get(dt_idx).position_)
                box_acc[score_idx][gt_idx] = diffnorm3(gt_boxes.get(gt_idx).dimension_, dt_boxes.get(dt_idx).dimension_)

                angular_acc_cur = (gt_boxes.get(gt_idx).orientation.inv() * dt_boxes.get(dt_idx).orientation).magnitude()
                angular_acc[score_idx][gt_idx] = angular_acc_cur / PI

                if dt_boxes[dt_idx].orientation_var > 0: # FIXME: these operations slow down the evaluator
                    var_acc_cur = sps.multivariate_normal.logpdf(gt_boxes[gt_idx].position,
                        dt_boxes[dt_idx].position, cov=dt_boxes[dt_idx].position_var)
                    var_acc_cur += sps.multivariate_normal.logpdf(gt_boxes[gt_idx].dimension,
                        dt_boxes[dt_idx].dimension, cov=dt_boxes[dt_idx].dimension_var)
                    var_acc_cur += sps.vonmises.logpdf(angular_acc_cur, kappa=1/dt_boxes[dt_idx].orientation_var)
                    var_acc[score_idx][gt_idx] = var_acc_cur
                else:
                    var_acc[score_idx][gt_idx] = -INFINITY

            # process detection match results
            for dt_idx in dt_indices:
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]     
                if matcher.query_src_match(dt_idx) < 0:
                    summary.fp[dt_tag][score_idx] += 1

        # aggregate accuracy metrics
        cdef vector[int] gt_tags
        gt_tags.reserve(gt_boxes.size())
        for gt_idx in range(gt_boxes.size()):
            gt_tags.push_back(gt_boxes.get(gt_idx).tag.labels[0])

        summary.acc_iou = self._aggregate_stats(iou_acc, gt_tags)
        summary.acc_angular = self._aggregate_stats(angular_acc, gt_tags)
        summary.acc_dist = self._aggregate_stats(dist_acc, gt_tags)
        summary.acc_box = self._aggregate_stats(box_acc, gt_tags)
        summary.acc_var = self._aggregate_stats(var_acc, gt_tags)
        return summary

    cpdef void add_stats(self, DetectionEvalStats stats) except*:
        '''
        Add statistics from calc_stats into database
        '''
        cdef int otp, ntp
        for k in self._classes:
            self._stats.ngt[k] += stats.ngt[k]
            for i in range(self._pr_nsamples):
                # aggregate accuracies
                otp, ntp = self._stats.tp[k][i], stats.tp[k][i]
                self._stats.acc_angular[k][i] = wmean(
                    self._stats.acc_angular[k][i], otp, stats.acc_angular[k][i], ntp)
                self._stats.acc_box[k][i] = wmean(
                    self._stats.acc_box[k][i], otp, stats.acc_box[k][i], ntp)
                self._stats.acc_iou[k][i] = wmean(
                    self._stats.acc_iou[k][i], otp, stats.acc_iou[k][i], ntp)
                self._stats.acc_dist[k][i] = wmean(
                    self._stats.acc_dist[k][i], otp, stats.acc_dist[k][i], ntp)
                self._stats.acc_var[k][i] = wmean(
                    self._stats.acc_var[k][i], otp, stats.acc_var[k][i], ntp)

                # aggregate common stats
                self._stats.ndt[k][i] = self._stats.ndt[k][i] + stats.ndt[k][i]
                self._stats.tp[k][i] = self._stats.tp[k][i] + stats.tp[k][i]
                self._stats.fp[k][i] = self._stats.fp[k][i] + stats.fp[k][i]
                self._stats.fn[k][i] = self._stats.fn[k][i] + stats.fn[k][i]

    cpdef DetectionEvalStats get_stats(self):
        '''
        Summarize current state of the benchmark counters
        '''
        return self._stats

    cdef inline int _get_score_idx(self, float score) nogil:
        if isnan(score):
            return self._pr_nsamples // 2
        else:
            return bisect(self._pr_thresholds, score)

    @property
    def score_thresholds(self):
        return np.asarray(self._pr_thresholds)

    def gt_count(self):
        return self._stats.ngt
    def dt_count(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.ndt}

    def tp(self, float score=NAN):
        '''Return true positive count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.tp}
    def fp(self, float score=NAN):
        '''Return false positive count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.fp}
    def fn(self, float score=NAN):
        '''Return false negative count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.fn}

    def precision(self, float score=NAN, bint return_all=False):
        cdef int score_idx
        if return_all:
            p = {self._class_type(k): [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    p[self._class_type(k)][i] = calc_precision(self._stats.tp[k][i], self._stats.fp[k][i])
        else:
            score_idx = self._get_score_idx(score)
            p = {self._class_type(k): calc_precision(self._stats.tp[k][score_idx], self._stats.fp[k][score_idx])
                 for k in self._classes}
        return p
    def recall(self, float score=NAN, bint return_all=False):
        cdef int score_idx
        if return_all:
            r = {self._class_type(k): [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    r[self._class_type(k)][i] = calc_recall(self._stats.tp[k][i], self._stats.fn[k][i])
        else:
            score_idx = self._get_score_idx(score)
            r = {self._class_type(k): calc_recall(self._stats.tp[k][score_idx], self._stats.fn[k][score_idx])
                 for k in self._classes}
        return r
    def fscore(self, float score=NAN, float beta=1, bint return_all=False):
        cdef float b2 = beta * beta
        cdef int score_idx
        if return_all:
            fs = {self._class_type(k): [None] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    fs[self._class_type(k)][i] = calc_fscore(self._stats.tp[k][i], self._stats.fp[k][i], self._stats.fn[k][i], b2)
        else:
            score_idx = self._get_score_idx(score)
            fs = {self._class_type(k): calc_fscore(self._stats.tp[k][score_idx], self._stats.fp[k][score_idx], self._stats.fn[k][score_idx], b2)
                for k in self._classes}
        return fs

    def ap(self):
        '''Calculate (mean) average precision'''
        p, r = self.precision(return_all=True), self.recall(return_all=True)
        # usually pr curve grows from bottom right to top left as score threshold
        # increases, so the area can be negative
        typed_classes = (self._class_type(k) for k in self._classes)
        area = {k: -np.trapz(p[k], r[k]) for k in typed_classes}
        return area

    def acc_iou(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.acc_iou}
    def acc_box(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.acc_box}
    def acc_dist(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.acc_dist}
    def acc_angular(self, float score=NAN):
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._stats.acc_angular}

    def summary(self, float score_thres = 0.8):
        '''
        Print default summary (into returned string)
        '''
        cdef int score_idx = self._get_score_idx(score_thres)

        cdef list lines = [''] # prepend an empty line
        precision, recall = self.precision(score_thres), self.recall(score_thres)
        fscore, ap = self.fscore(return_all=True), self.ap()

        lines.append("========== Benchmark Summary ==========")
        for k in self._classes:
            typed_k = self._class_type(k)
            lines.append("Results for %s:" % typed_k.name)
            lines.append("\tTotal processed targets:\t%d gt boxes, %d dt boxes" % (
                self._stats.ngt[k], max(self._stats.ndt[k])
            ))
            lines.append("\tPrecision (score > %.2f):\t%.3f" % (score_thres, precision[typed_k]))
            lines.append("\tRecall (score > %.2f):\t\t%.3f" % (score_thres, recall[typed_k]))
            lines.append("\tMax F1:\t\t\t\t%.3f" % max(fscore[typed_k]))
            lines.append("\tAP:\t\t\t\t%.3f" % ap[typed_k])
            lines.append("")
            lines.append("\tMean IoU (score > %.2f):\t\t%.3f" % (score_thres, self._stats.acc_iou[k][score_idx]))
            lines.append("\tMean angular error (score > %.2f):\t%.3f" % (score_thres, self._stats.acc_angular[k][score_idx]))
            lines.append("\tMean distance (score > %.2f):\t\t%.3f" % (score_thres, self._stats.acc_dist[k][score_idx]))
            lines.append("\tMean box error (score > %.2f):\t\t%.3f" % (score_thres, self._stats.acc_box[k][score_idx]))
            if not isinf(self._stats.acc_var[k][score_idx]):
                lines.append("\tMean variance error (score > %.2f):\t%.3f" % (score_thres, self._stats.acc_var[k][score_idx]))
        lines.append("========== Summary End ==========")

        return '\n'.join(lines)

ctypedef unsigned long long ull

@cython.auto_pickle(True)
cdef class TrackingEvalStats(DetectionEvalStats):
    ''' Tracking stats summary of a evaluation step '''

    cdef public unordered_map[int, vector[int]] id_switches
    ''' Number of tracked trajectory matched to different ground-truth trajectories '''
    
    cdef public unordered_map[int, vector[int]] fragments
    ''' Number of ground-truth trajectory matched to different tracked tracjetories '''

    cdef public unordered_map[int, unordered_map[ull, int]] ngt_ids
    ''' Frame count of all ground-truth targets (represented by their IDs) '''

    cdef public unordered_map[int, vector[unordered_map[ull, int]]] ngt_tracked
    ''' Frame count of ground-truth targets being tracked '''

    cdef public unordered_map[int, vector[unordered_map[ull, int]]] ndt_ids
    ''' Frame count of all proposal targets (represented by their IDs) '''

    cdef void initialize(self, unordered_set[int] classes, int nsamples):
        DetectionEvalStats.initialize(self, classes, nsamples)
        for k in classes:
            self.id_switches[k] = vector[int](nsamples, 0)
            self.fragments[k] = vector[int](nsamples, 0)

            self.ngt_ids[k] = unordered_map[ull, int]()
            self.ngt_tracked[k] = vector[unordered_map[ull, int]](nsamples)
            self.ndt_ids[k] = vector[unordered_map[ull, int]](nsamples)

    def as_object(self):
        ret = dict(ngt=self.ngt, tp=self.tp, fp=self.fp, fn=self.fn, ndt=self.ndt,
            acc_iou=self.acc_iou, acc_angular=self.acc_angular, acc_dist=self.acc_dist,
            acc_box=self.acc_box, acc_var=self.acc_var,
            id_switches=self.id_switches, fragments=self.fragments,
            ngt_ids={i.first: list(i.second) for i in self.ngt_ids},
            ngt_tracked={i.first: [list(s) for s in i.second] for i in self.ngt_tracked},
            ndt_ids={i.first: [list(s) for s in i.second] for i in self.ndt_ids}
        )

@cython.auto_pickle(True)
cdef class TrackingEvaluator(DetectionEvaluator):
    '''Benchmark for object tracking'''
    cdef TrackingEvalStats _tstats

    # temporary variables for tracking
    cdef vector[unordered_map[ull, ull]] _last_gt_assignment, _last_dt_assignment
    cdef vector[unordered_map[ull, int]] _last_gt_tags, _last_dt_tags

    def __init__(self, classes, min_overlaps, int pr_sample_count=40, float min_score=0, str pr_sample_scale="log10"):
        '''
        Object tracking benchmark. Targets association is done by score sorting.

        :param classes: Object classes to consider
        :param min_overlaps: Min overlaps per class for two boxes being considered as overlap.
            If single value is provided, all class will use the same overlap threshold
        :param min_score: Min score for precision-recall samples
        :param pr_sample_count: Number of precision-recall sample points (expect for p=1,r=0 and p=0,r=1)
        :param pr_sample_scale: PR sample type, {lin: linspace, log: logspace 1~10, logX: logspace 1~X}
        '''
        super().__init__(classes, min_overlaps, min_score=min_score,
            pr_sample_count=pr_sample_count, pr_sample_scale=pr_sample_scale)

        self._last_gt_assignment.resize(self._pr_nsamples)
        self._last_dt_assignment.resize(self._pr_nsamples)
        self._last_gt_tags.resize(self._pr_nsamples)
        self._last_dt_tags.resize(self._pr_nsamples)

        self._tstats = TrackingEvalStats()
        self._tstats.initialize(self._classes, self._pr_nsamples)
        self._stats = self._tstats

    cpdef void reset(self):
        DetectionEvaluator.reset(self)

        for k in self._classes:
            self._tstats.id_switches[k].assign(self._pr_nsamples, 0)
            self._tstats.fragments[k].assign(self._pr_nsamples, 0)

            self._tstats.ngt_ids[k].clear()
            for i in range(self._pr_nsamples):
                self._tstats.ngt_tracked[k][i].clear()

        for i in range(self._pr_nsamples):
            self._last_gt_assignment[i].clear()
            self._last_dt_assignment[i].clear()
            self._last_gt_tags[i].clear()
            self._last_dt_tags[i].clear()

    cpdef TrackingEvalStats calc_stats(self, Target3DArray gt_boxes, Target3DArray dt_boxes, TransformSet calib=None):
        # convert boxes to the same frame
        if gt_boxes.frame != dt_boxes.frame:
            if calib is None:
                raise ValueError("Calibration is not provided when dt_boxes and gt_boxes are in different frames!")
            dt_boxes = calib.transform_objects(dt_boxes, frame_to=gt_boxes.frame)

        # forward definitions
        cdef int gt_idx, gt_tag, dt_idx, dt_tag
        cdef ull dt_tid, gt_tid
        cdef float score_thres, angular_acc_cur, var_acc_cur
        cdef unordered_map[ull, int] gt_assignment_idx, dt_assignment_idx # store tid -> matched idx mapping
        cdef unordered_set[ull] gt_tid_set, dt_tid_set

        # initialize matcher
        cdef ScoreMatcher matcher = ScoreMatcher()
        matcher.prepare_boxes(dt_boxes, gt_boxes, DistanceTypes.RIoU)

        # initialize statistics
        cdef TrackingEvalStats summary = TrackingEvalStats()
        summary.initialize(self._classes, self._pr_nsamples)

        cdef vector[unordered_map[int, float]] iou_acc, angular_acc, dist_acc, box_acc, var_acc
        for k in self._classes:
            iou_acc.resize(self._pr_nsamples)
            angular_acc.resize(self._pr_nsamples)
            dist_acc.resize(self._pr_nsamples)
            box_acc.resize(self._pr_nsamples)
            var_acc.resize(self._pr_nsamples)

        
        # select ground-truth boxes to match
        cdef vector[int] gt_indices, dt_indices
        for gt_idx in range(gt_boxes.size()):
            gt_tag = gt_boxes.get(gt_idx).tag.labels[0]
            if self._classes.find(gt_tag) == self._classes.end():
                continue  # skip objects within ignored category

            gt_tid = gt_boxes.get(gt_idx).tid
            summary.ngt[gt_tag] += 1
            summary.ngt_ids[gt_tag][gt_tid] = 1
            gt_tid_set.insert(gt_tid)
            gt_indices.push_back(gt_idx)

        # loop over score thres
        for score_idx in range(self._pr_nsamples):
            score_thres = self._pr_thresholds[score_idx]

            # select detection boxes to match
            dt_indices.clear()
            dt_tid_set.clear()
            for dt_idx in range(dt_boxes.size()):
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]
                if self._classes.find(dt_tag) == self._classes.end():
                    continue  # skip objects within ignored category
                if dt_boxes.get(dt_idx).tag.scores[0] < score_thres:
                    continue  # skip objects with lower scores

                dt_tid = dt_boxes.get(dt_idx).tid
                assert dt_tid > 0, "Tracking id should be greater than 0 for a valid object!"
                dt_tid_set.insert(dt_tid)
                summary.ndt[dt_tag][score_idx] += 1
                summary.ndt_ids[dt_tag][score_idx][dt_tid] = 1

                if self._last_dt_assignment[score_idx].find(dt_tid) == self._last_dt_assignment[score_idx].end():
                    dt_indices.push_back(dt_idx)  # match objects without previous assignment
                else:
                    # preserve previous assignments as many as possible
                    gt_tid = self._last_dt_assignment[score_idx][dt_tid]
                    for gt_idx in range(gt_boxes.size()):
                        if gt_tid == gt_boxes.get(gt_idx).tid:  # find the gt boxes with stored tid
                            if matcher._distance_cache[dt_idx, gt_idx] > self._max_distance[dt_tag]:
                                dt_indices.push_back(dt_idx)  # also match objects that are apart from previous assignment
                            else:
                                gt_assignment_idx[gt_tid] = dt_idx
                                dt_assignment_idx[dt_tid] = gt_idx
                            break

            # match boxes
            matcher.clear_match()
            matcher.match(dt_indices, gt_indices, self._max_distance)

            # process ground-truth match results (gt_indices will always have all objects)
            for gt_idx in gt_indices:
                gt_tag = gt_boxes.get(gt_idx).tag.labels[0]
                gt_tid = gt_boxes.get(gt_idx).tid

                # update assignment
                dt_idx = matcher.query_dst_match(gt_idx)
                if dt_idx >= 0:
                    dt_tid = dt_boxes.get(dt_idx).tid
                    if gt_assignment_idx.find(gt_tid) != gt_assignment_idx.end():
                        # overwrite previous matching
                        dt_assignment_idx.erase(dt_boxes.get(gt_assignment_idx[gt_tid]).tid)
                        dt_tag = dt_boxes.get(dt_idx).tag.labels[0]
                        summary.fp[dt_tag][score_idx] += 1
                    gt_assignment_idx[gt_tid] = dt_idx
                    dt_assignment_idx[dt_tid] = gt_idx

                if gt_assignment_idx.find(gt_tid) == gt_assignment_idx.end():
                    summary.fn[gt_tag][score_idx] += 1
                    continue
                dt_idx = gt_assignment_idx[gt_tid]
                summary.tp[gt_tag][score_idx] += 1
                summary.ngt_tracked[gt_tag][score_idx][gt_tid] = 1

                # caculate accuracy values for various criteria
                iou_acc[score_idx][gt_idx] = 1 - matcher._distance_cache[dt_idx, gt_idx] # FIXME: not elegant here
                dist_acc[score_idx][gt_idx] = diffnorm3(gt_boxes.get(gt_idx).position_, dt_boxes.get(dt_idx).position_)
                box_acc[score_idx][gt_idx] = diffnorm3(gt_boxes.get(gt_idx).dimension_, dt_boxes.get(dt_idx).dimension_)

                angular_acc_cur = (gt_boxes.get(gt_idx).orientation.inv() * dt_boxes.get(dt_idx).orientation).magnitude()
                angular_acc[score_idx][gt_idx] = angular_acc_cur / PI

                if dt_boxes[dt_idx].orientation_var > 0:
                    var_acc_cur = sps.multivariate_normal.logpdf(gt_boxes[gt_idx].position,
                        dt_boxes[dt_idx].position, cov=dt_boxes[dt_idx].position_var)
                    var_acc_cur += sps.multivariate_normal.logpdf(gt_boxes[gt_idx].dimension,
                        dt_boxes[dt_idx].dimension, cov=dt_boxes[dt_idx].dimension_var)
                    var_acc_cur += sps.vonmises.logpdf(angular_acc_cur, kappa=1/dt_boxes[dt_idx].orientation_var)
                    var_acc[score_idx][gt_idx] = var_acc_cur
                else:
                    var_acc[score_idx][gt_idx] = -INFINITY

            # process detection match results
            for dt_idx in dt_indices:
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]
                dt_tid = dt_boxes.get(dt_idx).tid
                if dt_assignment_idx.find(dt_tid) == dt_assignment_idx.end():
                    summary.fp[dt_tag][score_idx] += 1

            # calculate id_switches and fragments
            for aiter in self._last_gt_assignment[score_idx]:
                gt_tid, dt_tid = aiter.first, aiter.second
                gt_tag = self._last_gt_tags[score_idx][gt_tid]
                if gt_assignment_idx.find(gt_tid) == gt_assignment_idx.end():
                    if gt_tid_set.find(gt_tid) != gt_tid_set.end():
                        summary.id_switches[gt_tag][score_idx] += 1
                elif dt_boxes.get(gt_assignment_idx[gt_tid]).tid != dt_tid:
                    summary.id_switches[gt_tag][score_idx] += 1

            for aiter in self._last_dt_assignment[score_idx]:
                dt_tid, gt_tid = aiter.first, aiter.second
                dt_tag = self._last_dt_tags[score_idx][dt_tid]
                if dt_assignment_idx.find(dt_tid) == dt_assignment_idx.end():
                    if dt_tid_set.find(dt_tid) != dt_tid_set.end():
                        summary.fragments[dt_tag][score_idx] += 1
                elif gt_boxes.get(dt_assignment_idx[dt_tid]).tid != gt_tid:
                    summary.fragments[dt_tag][score_idx] += 1

            # update assignment storage
            self._last_gt_assignment[score_idx].clear()
            self._last_dt_assignment[score_idx].clear()
            self._last_gt_tags[score_idx].clear()
            self._last_dt_tags[score_idx].clear()

            for giter in gt_assignment_idx:
                gt_tid, dt_idx = giter.first, giter.second
                dt_tag = dt_boxes.get(dt_idx).tag.labels[0]
                dt_tid = dt_boxes.get(dt_idx).tid
                gt_idx = dt_assignment_idx[dt_tid]
                gt_tag = gt_boxes.get(gt_idx).tag.labels[0]

                self._last_gt_assignment[score_idx][gt_tid] = dt_tid
                self._last_dt_assignment[score_idx][dt_tid] = gt_tid
                self._last_gt_tags[score_idx][gt_tid] = gt_tag
                self._last_dt_tags[score_idx][dt_tid] = dt_tag

            gt_assignment_idx.clear()
            dt_assignment_idx.clear()

        # aggregates accuracy metrics
        cdef vector[int] gt_tags
        gt_tags.reserve(gt_boxes.size())
        for gt_idx in range(gt_boxes.size()):
            gt_tags.push_back(gt_boxes.get(gt_idx).tag.labels[0])

        summary.acc_iou = self._aggregate_stats(iou_acc, gt_tags)
        summary.acc_angular = self._aggregate_stats(angular_acc, gt_tags)
        summary.acc_dist = self._aggregate_stats(dist_acc, gt_tags)
        summary.acc_box = self._aggregate_stats(box_acc, gt_tags)
        summary.acc_var = self._aggregate_stats(var_acc, gt_tags)
        return summary

    cpdef void add_stats(self, DetectionEvalStats stats) except*:
        DetectionEvaluator.add_stats(self, stats)
        cdef TrackingEvalStats tstats = <TrackingEvalStats> stats
        cdef ull gt_tid, dt_tid
        cdef int gt_count, dt_count

        for k in self._classes:
            for giter in tstats.ngt_ids[k]:
                gt_tid, gt_count = giter.first, giter.second
                if self._tstats.ngt_ids[k].find(gt_tid) == self._tstats.ngt_ids[k].end():
                    self._tstats.ngt_ids[k][gt_tid] = gt_count
                else:
                    self._tstats.ngt_ids[k][gt_tid] += gt_count

            for i in range(self._pr_nsamples):
                self._tstats.id_switches[k][i] += tstats.id_switches[k][i]
                self._tstats.fragments[k][i] += tstats.fragments[k][i]

                for giter in tstats.ngt_tracked[k][i]:
                    gt_tid, gt_count = giter.first, giter.second
                    if self._tstats.ngt_tracked[k][i].find(gt_tid) == self._tstats.ngt_tracked[k][i].end():
                        self._tstats.ngt_tracked[k][i][gt_tid] = gt_count
                    else:
                        self._tstats.ngt_tracked[k][i][gt_tid] += gt_count

                for diter in tstats.ndt_ids[k][i]:
                    dt_tid, dt_count = diter.first, diter.second
                    if self._tstats.ndt_ids[k][i].find(dt_tid) == self._tstats.ndt_ids[k][i].end():
                        self._tstats.ndt_ids[k][i][dt_tid] = dt_count
                    else:
                        self._tstats.ndt_ids[k][i][dt_tid] += dt_count

    cpdef TrackingEvalStats get_stats(self):
        '''
        Summarize current state of the benchmark counters
        '''
        return self._tstats

    def id_switches(self, float score=NAN):
        '''Return ID switch count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._tstats.id_switches}
    def fragments(self, float score=NAN):
        '''Return fragments count. If score is not specified, return the median value'''
        cdef int score_idx = self._get_score_idx(score)
        return {self._class_type(diter.first): diter.second[score_idx] for diter in self._tstats.fragments}
    def gt_traj_count(self):
        '''Return total ground-truth trajectory count. gt() will return total bounding box count'''
        return {self._class_type(diter.first): diter.second.size() for diter in self._tstats.ngt_ids}

    cdef dict _calc_frame_ratio(self, float score, float frame_ratio_threshold, bint high_pass, bint return_all):
        # helper function for tracked_ratio and lost_ratio
        cdef int score_idx
        cdef float frame_ratio
        if return_all:
            r = {k: [0] * self._pr_nsamples for k in self._classes}
            for k in self._classes:
                for i in range(self._pr_nsamples):
                    for diter in self._tstats.ngt_tracked[k][i]:
                        frame_ratio = float(diter.second) / self._tstats.ngt_ids[k][diter.first]
                        if high_pass and frame_ratio > frame_ratio_threshold:
                            r[k][i] += 1
                        if not high_pass and frame_ratio < frame_ratio_threshold:
                            r[k][i] += 1
            r = {self._class_type(k): [
                    float(v) / self._tstats.ngt_ids[k].size() for v in l
                ] for k, l in r.items()}
        else:
            score_idx = self._get_score_idx(score)
            r = {k: 0 for k in self._classes}
            for k in self._classes:
                for diter in self._tstats.ngt_tracked[k][score_idx]:
                    frame_ratio = float(diter.second) / self._tstats.ngt_ids[k][diter.first]
                    if high_pass and frame_ratio > frame_ratio_threshold:
                        r[k] += 1
                    if not high_pass and frame_ratio < frame_ratio_threshold:
                        r[k] += 1
            r = {self._class_type(k): float(v) / self._tstats.ngt_ids[k].size() for k, v in r.items()}
        return r

    def tracked_ratio(self, float score=NAN, float frame_ratio_threshold=0.8, bint return_all=False):
        '''
        Return the ratio of mostly tracked trajectories.

        :param frame_ratio_threshold: The threshold of ratio of tracked frames over total frames.
            A trajectory with higher tracked frames ratio will be counted as mostly tracked
        '''
        return self._calc_frame_ratio(score, frame_ratio_threshold, high_pass=True, return_all=return_all)

    def lost_ratio(self, float score=NAN, float frame_ratio_threshold=0.2, bint return_all=False):
        '''
        Return the ratio of mostly lost trajectories.

        :param frame_ratio_threshold: The threshold of ratio of tracked frames over total frames.
            A trajectory with lower tracked frames ratio will be counted as mostly tracked
        '''
        return self._calc_frame_ratio(score, frame_ratio_threshold, high_pass=False, return_all=return_all)

    def mota(self, float score=NAN):
        '''Return the MOTA metric defined by the CLEAR MOT metrics. For MOTP equivalents, see acc_* properties'''
        # TODO: is this yielding negative value correct?
        cdef int score_idx = self._get_score_idx(score)
        ret = {self._class_type(k): 1 - float(self._stats.fp[k][score_idx] + self._stats.fn[k][score_idx] + self._tstats.id_switches[k][score_idx])
            / self._stats.ngt[k] for k in self._classes}
        return ret

    def summary(self, float score_thres = 0.8, float tracked_ratio_thres = 0.8, float lost_ratio_thres = 0.2, str note = None):
        '''
        Print default summary (into returned string)
        '''
        cdef int score_idx = self._get_score_idx(score_thres)

        cdef list lines = [''] # prepend an empty line
        precision, recall = self.precision(score_thres), self.recall(score_thres)
        fscore, ap = self.fscore(return_all=True), self.ap()

        mlt = self.tracked_ratio(score_thres, tracked_ratio_thres)
        mll = self.lost_ratio(score_thres, lost_ratio_thres)
        mota = self.mota(score_thres)

        if note:
            lines.append("========== Benchmark Summary (%s) ==========" % note)
        else:
            lines.append("========== Benchmark Summary ==========")
        for k in self._classes:
            typed_k = self._class_type(k)
            lines.append("Results for %s:" % typed_k.name)
            lines.append("\tTotal processed targets:\t%d gt boxes, %d dt boxes" % (
                self._stats.ngt[k], max(self._stats.ndt[k])
            ))
            lines.append("\tTotal processed trajectories:\t%d gt tracklets, %d dt tracklets" % (
                self.gt_traj_count()[typed_k], max(len(self._tstats.ndt_ids[k][i]) for i in range(self._pr_nsamples))
            ))
            lines.append("\tPrecision (score > %.2f):\t%.3f" % (score_thres, precision[typed_k]))
            lines.append("\tRecall (score > %.2f):\t\t%.3f" % (score_thres, recall[typed_k]))
            lines.append("\tMax F1:\t\t\t\t%.3f" % max(fscore[typed_k]))
            lines.append("\tAP:\t\t\t\t%.3f" % ap[typed_k])
            lines.append("")
            lines.append("\tID switches (score > %.2f):\t\t\t%d" % (score_thres, self._tstats.id_switches[k][score_idx]))
            lines.append("\tFragments (score > %.2f):\t\t\t%d" % (score_thres, self._tstats.fragments[k][score_idx]))
            lines.append("\tMOTA (score > %.2f):\t\t\t\t%.2f" % (score_thres, mota[typed_k]))
            lines.append("\tMostly tracked (score > %.2f, ratio > %.2f):\t%.3f" % (
                score_thres, tracked_ratio_thres, mlt[typed_k]))
            lines.append("\tMostly lost (score > %.2f, ratio < %.2f):\t%.3f" % (
                score_thres, lost_ratio_thres, mll[typed_k]))
            lines.append("")
            lines.append("\tMean IoU (score > %.2f):\t\t%.3f" % (score_thres, self._stats.acc_iou[k][score_idx]))
            lines.append("\tMean angular error (score > %.2f):\t%.3f" % (score_thres, self._stats.acc_angular[k][score_idx]))
            lines.append("\tMean distance (score > %.2f):\t\t%.3f" % (score_thres, self._stats.acc_dist[k][score_idx]))
            lines.append("\tMean box error (score > %.2f):\t\t%.3f" % (score_thres, self._stats.acc_box[k][score_idx]))
            if not isinf(self._stats.acc_var[k][score_idx]):
                lines.append("\tMean variance error (score > %.2f):\t%.3f" % (score_thres, self._stats.acc_var[k][score_idx]))
        lines.append("========== Summary End ==========")

        return '\n'.join(lines)
