from enum import Enum
from typing import List, Dict
from d3d.abstraction import Target3DArray

class DistanceTypes(Enum):
    IoU: int
    RIoU: int
    Position: int

class BaseMatcher:
    def clear_match(self): ...
    def prepare_boxes(self,
                      src_boxes: Target3DArray,
                      dst_boxes: Target3DArray,
                      distance_metric: DistanceTypes): ...
    def match(self,
              src_subset: List[int],
              dst_subset: List[int],
              distance_threshold: Dict[int, float]): ...
    def match_by_order(self,
                       src_order: List[int],
                       dst_order: List[int],
                       distance_threshold: Dict[int, float]): ...
    def query_src_match(self, src_idx: int) -> int: ...
    def query_dst_match(self, dst_idx: int) -> int: ...
    def num_of_matches(self) -> int: ...

class ScoreMatcher(BaseMatcher):
    pass

class NearestNeighborMatcher(BaseMatcher):
    pass

class HungarianMatcher(BaseMatcher):
    pass
