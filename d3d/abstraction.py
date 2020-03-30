import enum
import numpy as np
import logging
from scipy.spatial.transform import Rotation

_logger = logging.getLogger("d3d")

class ObjectTag:
    '''
    This class stands for label tags associate with object target
    '''
    def __init__(self, labels, mapping, scores=None):
        if not issubclass(mapping, enum.Enum):
            raise ValueError("The object class mapping should be an Enum")
        self.mapping = mapping

        # sanity check
        if scores is None:
            if isinstance(labels, (list, tuple)):
                raise ValueError("There cannot be multiple labels without scores")
            self.labels = [labels]
            self.scores = [1]
        else:
            if not isinstance(labels, (list, tuple)):
                self.labels = [labels]
                self.scores = [scores]
            else:
                self.labels = labels
                self.scores = scores

        # convert labels to enum object
        for i in range(len(self.labels)):
            if isinstance(self.labels[i], str):
                self.labels[i] = self.mapping[self.labels[i]]
            elif isinstance(self.labels[i], int):
                self.labels[i] = self.mapping(self.labels[i])
                
        # sort labels descending
        order = list(reversed(np.argsort(self.scores)))
        self.labels = [self.labels[i] for i in order]
        self.scores = [self.scores[i] for i in order]

    def __str__(self):
        return "<ObjectTag, top class: %s>" % self.labels[0].name

class ObjectTarget3D:
    '''
    This class stands for a target in cartesian coordinate. The body coordinate is FLU (front-left-up).
    '''
    def __init__(self, position, orientation, dimension, tag, id=None):
        '''
        :param position: Position of object center (x,y,z)
        :param orientation: Object heading (direction of x-axis attached on body)
            with regard to x-axis of the world at the object center.
        :param dimension: Length of the object in 3 dimensions (lx,ly,lz)
        :param tag: Classification information of the object
        :param id: ID of the object used for tracking (optional)
        '''

        assert len(position) == 3, "Invalid position shape"
        self.position = np.array(position)

        assert len(dimension) == 3, "Invalid dimension shape"
        self.dimension = np.array(dimension)

        if isinstance(orientation, Rotation):
            self.orientation = orientation
        elif len(orientation) == 4:
            self.orientation = Rotation.from_quat(orientation)
        else:
            raise ValueError("Invalid rotation format")

        if isinstance(tag, ObjectTag):
            self.tag = tag
        else:
            raise ValueError("Label should be of type ObjectTag")

        self.id = id

    @property
    def tag_name(self):
        return self.tag.labels[0].name

    @property
    def tag_score(self):
        return self.tag.scores[0]

    @property
    def yaw(self):
        '''
        Return the rotation angle around z-axis (ignoring rotations in other two directions)
        '''
        angles = self.orientation.as_euler("ZYX")
        if abs(angles[1]) + abs(angles[2]) > 0.1:
            _logger.warn("The roll (%.2f) and pitch(%.2f) angle in objects may be to large to ignore!",
                angles[2], angles[1])
        return angles[0]

class ObjectTarget3DArray(list):
    def __init__(self, iterable=[]):
        super().__init__(iterable)

    def to_numpy(self, box_type="ground"):
        '''
        :param box_type: Decide how to represent the box
        '''
        if len(self) == 0:
            return np.empty((0, 8))

        def to_ground(box):
            cls_value = box.tag.labels[0].value
            arr = np.concatenate([box.position, box.dimension, [box.yaw, cls_value]])
            return arr # store only 3D box and label
        return np.stack([to_ground(box) for box in self])

    def to_torch(self, box_type="ground"):
        import torch
        return torch.tensor(self.to_numpy(), box_type=box_type)

    def to_kitti(self):
        pass

    def __str__(self):
        return "<ObjectTarget3DArray with %d objects>" % len(self)

class ParameterSet:
    '''
    This object load a collection of intrinsic and extrinsic parameters
    '''
    def __init__(self):
        self.intrinsics = {}
        self.extrinsics = {}
        
    def add_intrinsics(self, frame_id, projection_matrix):
        pass

    def add_extrinsics(self, frame_from, frame_to, projection_matrix):
        pass
