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
    def __init__(self, position, orientation, dimension, tag):
        '''
        :param position: Position of object center (x,y,z)
        :param orientation: Object heading (direction of x-axis attached on body)
            with regard to x-axis of the world at the object center.
        :param dimension: Length of the object in 3 dimensions (lx,ly,lz)
        :param tag: Classification information of the object
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

    def to_numpy(self):
        pass

    def to_kitti(self):
        pass

    def __str__(self):
        return "<ObjectTarget3DArray with %d objects>" % len(self)
