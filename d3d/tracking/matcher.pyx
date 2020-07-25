import torch
cimport numpy as np
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from d3d.box import box2d_iou

cdef class BaseMatcher:
    cpdef void prepare_boxes(self, ObjectTarget3DArray src_boxes, ObjectTarget3DArray dst_boxes, DistanceTypes distance_metric):
        '''
        This method add two arrays of boxes and prepare related informations, it will also clean previous
        results.

        :param src_boxes: boxes to match
        :param dst_boxes: fixed boxes (such as ground truth boxes)
        '''
        self._src_boxes = src_boxes
        self._dst_boxes = dst_boxes

        # sometimes pre-calculate these values will be slower?
        cdef np.ndarray src_arr = src_boxes.to_numpy().astype(np.float32)
        cdef np.ndarray dst_arr = dst_boxes.to_numpy().astype(np.float32)
        if distance_metric == DistanceTypes.IoU:
            self._distance_cache = 1 - box2d_iou( # use 1-iou as distance
                src_arr[:, [0,1,3,4,6]],
                dst_arr[:, [0,1,3,4,6]],
                method="box") 
        if distance_metric == DistanceTypes.RIoU:
            self._distance_cache = 1 - box2d_iou(
                src_arr[:, [0,1,3,4,6]],
                dst_arr[:, [0,1,3,4,6]],
                method="rbox")
        elif distance_metric == DistanceTypes.Position:
            self._distance_cache = cdist(src_arr[:, :3], dst_arr[:, :3], metric='euclidean').astype(np.float32)

    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        '''
        :param src_subset: Indices of source boxes to be considered
        :param dst_subset: Indices of destination boxes to be considered
        :param distance_threshold: threshold of maximum distance for each category. The category should be represented as its value.
        '''
        raise NotImplementedError("This is a virtual function!")

    cdef void match_by_order(self, vector[int]& src_order, vector[int]& dst_order, unordered_map[int, float]& distance_threshold):
        '''
        Match boxes with given order (src-dst pairs). Make sure src_order and dst_order has same size.
        '''
        assert src_order.size() == dst_order.size(), "The sizes of src and dst order vector should be the same!"
        for i in range(src_order.size()):
            src_idx = src_order[i]
            dst_idx = dst_order[i]

            # skip already assigned box
            if self._src_assignment.find(src_idx) != self._src_assignment.end():
                continue
            if self._dst_assignment.find(dst_idx) != self._dst_assignment.end():
                continue

            # compare category information
            src_tag = self._src_boxes.get(src_idx).tag.labels[0]
            dst_tag = self._dst_boxes.get(dst_idx).tag.labels[0]
            if src_tag != dst_tag:
                continue

            # true positive if distance is smaller than threshold
            if self._distance_cache[src_idx, dst_idx] <= distance_threshold[dst_tag]:
                self._src_assignment[src_idx] = dst_idx
                self._dst_assignment[dst_idx] = src_idx

            # if all src or dst boxes are matched, the process is finished
            if self._src_assignment.size() == src_order.size():
                break
            if self._dst_assignment.size() == src_order.size():
                break

    cpdef int query_src_match(self, int src_idx):
        if self._src_assignment.find(src_idx) == self._src_assignment.end():
            return -1
        return self._src_assignment[src_idx]

    cpdef int query_dst_match(self, int dst_idx):
        if self._dst_assignment.find(dst_idx) == self._dst_assignment.end():
            return -1
        return self._dst_assignment[dst_idx]

    cpdef int num_of_matches(self):
        assert self._src_assignment.size() == self._dst_assignment.size()
        return self._src_assignment.size()

cdef class ScoreMatcher:
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        self._src_assignment.clear()
        self._dst_assignment.clear()

        # sort src boxes by their score, and sort dst boxes by its distance to src boxes
        cdef list src_list = src_subset, dst_list = dst_subset
        cdef list src_scores = [self._src_boxes.get(sidx).tag.scores[0] for sidx in src_subset]
        cdef np.ndarray[long, ndim=1] src_order = np.flip(np.argsort(src_scores)) # match from best score
        cdef np.ndarray distance_subset = np.asarray(self._distance_cache)[np.ix_(src_list, dst_list)]
        cdef np.ndarray[long, ndim=2] dst_order = np.argsort(distance_subset, axis=1)
        
        # reform into vectors
        cdef vector[int] src_indices, dst_indices
        cdef int npairs = src_subset.size() * dst_subset.size()
        src_indices.reserve(npairs)
        dst_indices.reserve(npairs)
        
        for src_idx in range(src_subset.size()):
            for dst_idx in range(dst_subset.size()):
                src_indices.push_back(src_subset[src_order[src_idx]])
                dst_indices.push_back(dst_subset[dst_order[src_idx, dst_idx]])

        # actual matching
        self.match_by_order(src_indices, dst_indices, distance_threshold)

cdef class NearestNeighborMatcher:
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        self._src_assignment.clear()
        self._dst_assignment.clear()

        # sort the match pairs by distance
        cdef list src_list = src_subset, dst_list = dst_subset
        cdef np.ndarray distance_subset = np.asarray(self._distance_cache)[np.ix_(src_list, dst_list)]
        cdef np.ndarray distance_order = np.argsort(distance_subset, axis=None)

        # reform the match pairs into vectors
        cdef np.ndarray[long, ndim=1] sorted_src_indices, sorted_dst_indices
        sorted_src_indices, sorted_dst_indices = np.unravel_index(distance_order, (src_subset.size(), dst_subset.size()))

        cdef vector[int] src_indices, dst_indices
        src_indices.reserve(sorted_src_indices.size)
        dst_indices.reserve(sorted_dst_indices.size)
        for i in range(sorted_src_indices.size):
            src_indices.push_back(src_subset[sorted_src_indices[i]])
            dst_indices.push_back(dst_subset[sorted_dst_indices[i]])

        # actual matching
        self.match_by_order(src_indices, dst_indices, distance_threshold)

cdef class HungarianMatcher:
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        # split the input by categories
        cdef dict src_classes = {}, dst_classes = {}
        cdef int src_idx, dst_idx

        for src_idx in src_subset:
            src_tag = self._src_boxes.get(src_idx).tag.labels[0]
            if src_tag in src_classes:
                src_classes[src_tag].append(src_idx)
            else:
                src_classes[src_tag] = [src_idx]
        for dst_idx in dst_subset:
            dst_tag = self._dst_boxes.get(dst_idx).tag.labels[0]
            if dst_tag in dst_classes:
                dst_classes[dst_tag].append(dst_idx)
            else:
                dst_classes[dst_tag] = [dst_idx]
            
        # forward definitions
        cdef list src_list, dst_list
        cdef np.ndarray distance_subset
        cdef np.ndarray[long, ndim=1] src_optim_indices, dst_optim_indices

        for clsid in src_classes.keys():
            if clsid not in dst_classes.keys():
                continue # only consider common categories

            # extract submatrix
            src_list = src_classes[clsid]
            dst_list = dst_classes[clsid]
            distance_subset = np.asarray(self._distance_cache)[np.ix_(src_list, dst_list)]

            # optimize using scipy
            src_optim_indices, dst_optim_indices = linear_sum_assignment(distance_subset)
            
            # store the results and apply threshold
            for i in range(src_optim_indices.size):
                src_idx = src_list[src_optim_indices[i]]
                dst_idx = dst_list[dst_optim_indices[i]]
                if self._distance_cache[src_idx, dst_idx] <= distance_threshold[clsid]:
                    self._src_assignment[src_idx] = dst_idx
                    self._dst_assignment[dst_idx] = src_idx
