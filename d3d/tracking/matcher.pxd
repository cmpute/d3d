from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from d3d.abstraction cimport ObjectTarget3D, Target3DArray

cpdef enum DistanceTypes:
    IoU = 1 # Box IoU
    RIoU = 2 # Rotated box IoU
    Position = 3 # Euclidean distance on position

cdef class BaseMatcher:
    cdef Target3DArray _src_boxes, _dst_boxes
    cdef float[:, :] _distance_cache
    cdef unordered_map[int, int] _src_assignment, _dst_assignment

    cpdef void clear_match(self)
    cpdef void prepare_boxes(self, Target3DArray src_boxes, Target3DArray dst_boxes, DistanceTypes distance_metric)
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)
    cdef void match_by_order(self, vector[int]& src_order, vector[int]& dst_order, unordered_map[int, float]& distance_threshold)
    cpdef int query_src_match(self, int src_idx)
    cpdef int query_dst_match(self, int dst_idx)
    cpdef int num_of_matches(self)

cdef class ScoreMatcher(BaseMatcher):
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)

cdef class NearestNeighborMatcher(BaseMatcher):
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)

cdef class HungarianMatcher(BaseMatcher):
    cpdef void match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)
