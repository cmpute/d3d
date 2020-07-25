from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from d3d.abstraction cimport ObjectTarget3D, ObjectTarget3DArray

cpdef enum DistanceTypes:
    IoU = 1
    RIoU = 2
    Euclidean = 3

cdef class BaseMatcher:
    cdef ObjectTarget3DArray _src_boxes, _dst_boxes
    cdef float[:, :] _distance_cache
    cdef unordered_map[int, int] _src_assignment, _dst_assignment

    cpdef prepare_boxes(self, ObjectTarget3DArray src_boxes, ObjectTarget3DArray dst_boxes, DistanceTypes distance_metric)
    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)
    cdef void match_by_order(self, vector[int]& src_order, vector[int]& dst_order, unordered_map[int, float]& distance_threshold)
    cpdef int query_src_match(self, int src_idx)
    cpdef int query_dst_match(self, int dst_idx)
    cpdef int num_of_matches(self)

cdef class ScoreMatcher(BaseMatcher):
    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)

cdef class NearestNeighborMatcher(BaseMatcher):
    cpdef match(self, vector[int] src_subset, vector[int] dst_subset, unordered_map[int, float] distance_threshold)
