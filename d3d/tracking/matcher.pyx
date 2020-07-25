import torch
cimport numpy as np
import numpy as np

from scipy.spatial.distance import cdist
from d3d.box import box2d_iou

cdef class BaseMatcher:
    cpdef prepare_boxes(self, ObjectTarget3DArray src_boxes, ObjectTarget3DArray dst_boxes, DistanceTypes distance_metric):
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
            self._distance_cache = -box2d_iou( # use negative value as distance
                src_arr[:, [0,1,3,4,6]],
                dst_arr[:, [0,1,3,4,6]],
                method="box") 
        if distance_metric == DistanceTypes.RIoU:
            self._distance_cache = -box2d_iou( # use negative value as distance
                src_arr[:, [0,1,3,4,6]],
                dst_arr[:, [0,1,3,4,6]],
                method="rbox")
        elif distance_metric == DistanceTypes.Euclidean:
            self._distance_cache = cdist(src_arr[:, :3], dst_arr[:, :3], metric='euclidean').astype(np.float32)

    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
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
        for i in range(src_order.size()):
            src_idx = src_order[i]
            dst_idx = dst_order[i]

            # compare category information
            src_tag = self._src_boxes.get(src_idx).tag.labels[0]
            dst_tag = self._dst_boxes.get(dst_idx).tag.labels[0]
            if src_tag != dst_tag:
                continue

            # skip already assigned box
            if self._src_assignment.find(src_idx) != self._src_assignment.end():
                continue
            if self._dst_assignment.find(dst_idx) != self._dst_assignment.end():
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
        idx = self._src_assignment.find(src_idx)
        if idx == self._src_assignment.end():
            return -1
        return self._src_assignment[src_idx]

    cpdef int query_dst_match(self, int dst_idx):
        idx = self._dst_assignment.find(dst_idx)
        if idx == self._dst_assignment.end():
            return -1
        return self._dst_assignment[dst_idx]

    cpdef int num_of_matches(self):
        assert self._src_assignment.size() == self._dst_assignment.size()
        return self._src_assignment.size()

cdef class ScoreMatcher:
    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        self._src_assignment.clear()
        self._dst_assignment.clear()
        cdef list src_scores = [self._src_boxes.get(sidx).tag.scores[0] for sidx in src_subset]
        cdef vector[long] src_order = np.argsort(src_scores)[::-1].tolist() # match from best score

        for dst_idx in dst_subset:
            dst_tag = self._dst_boxes.get(dst_idx).tag.labels[0]

            # skip already assigned destination box
            if self._dst_assignment.find(dst_idx) != self._dst_assignment.end():
                continue
            
            for order_idx in src_order:
                src_idx = src_subset[order_idx]

                # compare category information
                if self._src_boxes.get(src_idx).tag.labels[0] != dst_tag:
                    continue

                # skip already assigned source box
                if self._src_assignment.find(src_idx) != self._src_assignment.end():
                    continue

                # true positive if distance is smaller than threshold
                if self._distance_cache[src_idx, dst_idx] <= distance_threshold[dst_tag]:
                    self._src_assignment[src_idx] = dst_idx
                    self._dst_assignment[dst_idx] = src_idx

                # if all src or dst boxes are matched, the process is finished
                if self._src_assignment.size() == src_subset.size():
                    return
                if self._dst_assignment.size() == dst_subset.size():
                    return

cdef class NearestNeighborMatcher:
    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        self._src_assignment.clear()
        self._dst_assignment.clear()

        # sort the match pairs by distance
        cdef list src_list = src_subset, dst_list = dst_subset
        distance_subset = np.asarray(self._distance_cache)[src_list, :][:, dst_list] # not using cdef here since ndarray.shape returns npy_intp*
        distance_order = np.argsort(distance_subset, axis=None)

        # reform the match pairs into vectors
        sorted_src_indices, sorted_dst_indices = np.unravel_index(distance_order, distance_subset.shape)
        cdef vector[int] src_indices = sorted_src_indices.tolist()
        cdef vector[int] dst_indices = sorted_dst_indices.tolist()
        self.match_by_order(src_indices, dst_indices, distance_threshold)
