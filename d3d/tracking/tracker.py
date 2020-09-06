
import numpy as np

from d3d.abstraction import ObjectTarget3D, Target3DArray, TrackingTarget3D
from d3d.tracking.filter import Pose_3DOF_UKF_CTRA, Box_KF
from d3d.tracking.matcher import HungarianMatcher, DistanceTypes

class VanillaTracker:
    def __init__(self,
        pose_tracker_factory=Pose_3DOF_UKF_CTRA,
        feature_tracker_factory=Box_KF,
        matcher_factory=HungarianMatcher,
        matcher_distance_type="position",
        matcher_distance_threshold=1,
        lost_time=1,
        default_position_var=np.eye(3),
        default_dimension_var=np.eye(3),
        default_orientation_var=1
    ):
        '''
        :param lost_time: determine the time length of a target being lost before it's removed from tracking
        :param pose_tracker_factory: factory function to generate a new pose tracker, takes only initial detection as input
        :param feature_tracker_factory: factory function to generate a new feature tracker, takes only initial detection as input
        :param matcher_factory: factory function to generate a new target matcher
        :param matcher_distance_type: distance type used to match targets
        :param matcher_distance_threshold: distance threshold used in target matcher
        :param default_position_var: default positional covariance assigned to targets (if not provided)
        :param default_dimension_var: default dimensional covariance assigned to targets (if not provided)
        :param default_orientation_var: default orientational covariance assigned to targets (if not provided)
        '''
        self._tracked_poses = dict() # Object trackers
        self._tracked_features = dict() # Feature trackers (shape, class, etc)
        self._timer_track = dict() # Timer for frames tracked consecutively
        self._timer_lost = dict() # Timer for frames lost consecutively

        self._default_position_var = default_position_var
        self._default_dimension_var = default_dimension_var
        self._default_orientation_var = default_orientation_var

        self._last_timestamp = None
        self._last_frameid = None
        self._id_counter = 1 # Counter for id generation of tracked objects, starting from 0 since 0 is considered as no id
        self._lost_time = lost_time

        self._pose_factory = pose_tracker_factory
        self._feature_factory = feature_tracker_factory
        self._matcher = matcher_factory()

        matcher_mapping = {
            "iou": DistanceTypes.IoU,
            "riou": DistanceTypes.RIoU,
            "position": DistanceTypes.Position
        }
        if isinstance(matcher_distance_type, str):
            self._match_distance = matcher_mapping[matcher_distance_type.lower()]
        else:
            self._match_distance = matcher_distance_type
        self._match_threshold = matcher_distance_threshold

    def _initialize(self, target: ObjectTarget3D):
        '''
        Initialize a new target
        '''
        self._tracked_poses[self._id_counter] = self._pose_factory(target)
        self._tracked_features[self._id_counter] = self._feature_factory(target)
        self._timer_track[self._id_counter] = 0.
        self._timer_lost[self._id_counter] = 0.
        self._id_counter += 1

    @property
    def tracked_ids(self):
        '''
        Return a ID list of actively tracked targets
        '''
        return list(self._tracked_poses.keys())

    def _current_objects_array(self) -> Target3DArray:
        '''
        Create Target3DArray from current tracked objects
        '''
        array = Target3DArray(frame=self._last_frameid, timestamp=self._last_timestamp)
        for tid in self.tracked_ids:
            target = ObjectTarget3D(
                position=self._tracked_poses[tid].position,
                orientation=self._tracked_poses[tid].orientation,
                dimension=self._tracked_features[tid].dimension,
                tag=self._tracked_features[tid].classification,
                tid=tid,
                position_var=self._tracked_poses[tid].position_var,
                orientation_var=self._tracked_poses[tid].orientation_var,
                dimension_var=self._tracked_features[tid].dimension_var
            )
            array.append(target)
        return array

    def _assign_default_var(self, target: ObjectTarget3D):
        if not np.any(target.position_var):
            target.position_var = self._default_position_var
        if not np.any(target.dimension_var):
            target.dimension_var = self._default_dimension_var
        if not np.any(target.orientation_var):
            target.orientation_var = self._default_orientation_var
        return target

    def update(self, detections: Target3DArray):
        '''
        Update the filters when receiving new detections
        '''
        if self._last_timestamp is None:
            # Initialize all trackers
            for target in detections:
                self._assign_default_var(target)
                self._initialize(target)
        else:
            # do prediction on each tracklet
            dt = detections.timestamp - self._last_timestamp
            for tracker in self._tracked_poses.values():
                tracker.predict(dt)
            for tracker in self._tracked_features.values():
                tracker.predict(dt)

            # Match existing trackers
            current_targets = self._current_objects_array()
            
            if isinstance(self._match_threshold, (float, int)):
                thresholds = {box.tag_top.value: float(self._match_threshold)
                    for box in (current_targets + detections)}
            else:
                assert isinstance(self._match_threshold, dict)
                thresholds = self._match_threshold
            self._matcher.prepare_boxes(detections, current_targets, self._match_distance)
            self._matcher.match(
                list(range(len(detections))),
                list(range(len(current_targets))),
                thresholds
            )

            lost_indices = set(self.tracked_ids)
            for idx, target in enumerate(detections):
                idx_match = self._matcher.query_src_match(idx)
                self._assign_default_var(target)
                if idx_match < 0:
                    # Initialize this detector
                    self._initialize(target)
                else:
                    tid = current_targets[idx_match].tid
                    self._tracked_poses[tid].update(target)
                    self._tracked_features[tid].update(target)
                    self._timer_lost[tid] = 0.
                    self._timer_track[tid] += dt

                    if tid in lost_indices:
                        lost_indices.remove(tid)

            for idx in lost_indices:
                self._timer_lost[idx] += dt
                self._timer_track[idx] = 0.

        # Deal with out-dated or invalid trackers
        rm_list = []
        for tid, time in self._timer_lost.items():
            if time > self._lost_time: # XXX: also remove if variance is too large
                rm_list.append(tid)
        for idx in rm_list:
            del self._tracked_poses[idx]
            del self._tracked_features[idx]
            del self._timer_lost[idx]
            del self._timer_track[idx]

        # Update times
        self._last_timestamp = detections.timestamp
        self._last_frameid = detections.frame

    def report(self) -> Target3DArray:
        '''
        Return the collection of valid tracked targets
        '''
        array = Target3DArray(frame=self._last_frameid, timestamp=self._last_timestamp)
        for tid in self.tracked_ids:
            target = TrackingTarget3D(
                position=self._tracked_poses[tid].position,
                orientation=self._tracked_poses[tid].orientation,
                dimension=self._tracked_features[tid].dimension,
                velocity=self._tracked_poses[tid].velocity,
                angular_velocity=self._tracked_poses[tid].angular_velocity,
                tag=self._tracked_features[tid].classification,
                tid=tid,
                position_var=self._tracked_poses[tid].position_var,
                orientation_var=self._tracked_poses[tid].orientation_var,
                dimension_var=self._tracked_features[tid].dimension_var,
                velocity_var=self._tracked_poses[tid].velocity_var,
                angular_velocity_var=self._tracked_poses[tid].angular_velocity_var,
                history=self._timer_track[tid] # TODO: discount the object score by lost time
            )
            array.append(target)

        return array

    @property
    def match_count(self):
        return self._matcher.num_of_matches()
