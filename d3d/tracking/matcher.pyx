import torch
cimport numpy as np
import numpy as np

from d3d.box import box2d_iou

cdef class ScoreMatcher:
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
            pass # TODO: implement

    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold):
        '''
        :param src_subset: Indices of source boxes to be considered
        :param dst_subset: Indices of destination boxes to be considered
        :parma distance_threshold: threshold of maximum distance for
        '''
        self._src_assignment.clear()
        self._dst_assignment.clear()
        cdef vector[long] src_order = np.argsort([self._src_boxes[sidx].tag_score for sidx in src_subset])[::-1].tolist() # match from best score

        for dst_idx in dst_subset:            
            dst_tag = self._dst_boxes[dst_idx].tag_top
            dst_tagv = dst_tag.value

            # skip already assigned box
            if self._dst_assignment.find(dst_idx) != self._dst_assignment.end():
                continue
            
            for order_idx in src_order:
                src_idx = src_subset[order_idx]

                # compare class information
                if self._src_boxes[src_idx].tag_top != dst_tag:
                    continue

                # skip already assigned source box
                if self._src_assignment.find(src_idx) != self._src_assignment.end():
                    continue

                # true positive if overlap is larger than threshold
                if self._distance_cache[src_idx, dst_idx] <= distance_threshold[dst_tagv]:
                    self._src_assignment[src_idx] = dst_idx
                    self._dst_assignment[dst_idx] = src_idx

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
