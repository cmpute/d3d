import enum
import logging
from collections import namedtuple

import numpy as np
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
        if hasattr(self.labels[0], "uname"):
            name = self.labels[0].uname
        else:
            name = self.labels[0].name
        return "<ObjectTag, top class: %s>" % name

class ObjectTarget3D:
    '''
    This class stands for a target in cartesian coordinate. The body coordinate is FLU (front-left-up).
    '''
    def __init__(self, position, orientation, dimension, tag,
        id=None, position_var=None, orientation_var=None, dimension_var=None):
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
        self.position_var = position_var or np.zeros((3, 3))
        self.dimension_var = dimension_var or np.zeros((3, 3))
        self.orientation_var = orientation_var or 0

    @property
    def tag_top(self):
        return self.tag.labels[0]

    @property
    def tag_name(self):
        '''
        Return the name of the target's top tag
        '''
        return self.tag_top.uname if hasattr(self.tag_top, "uname") else self.tag_top.name

    @property
    def tag_score(self):
        '''
        Return the score of the target's top tag
        '''
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

    @property
    def corners(self):
        '''
        Convert the bounding box representation to coorindate of 8 corner points
        '''
        offsets = [[-d/2, d/2] for d in self.dimension]
        offsets = np.array(np.meshgrid(*offsets)).T.reshape(-1, 3)
        offsets = offsets.dot(self.orientation.as_matrix().T)
        return self.position + offsets

class ObjectTarget3DArray(list):
    def __init__(self, iterable=[], frame=None):
        '''
        :param frame: Frame that the box parameters used. None means base frame (in TransformSet)
        '''
        super().__init__(iterable)
        self.frame = frame

        # copy frame value
        if isinstance(iterable, ObjectTarget3DArray) and not frame:
            self.frame = iterable.frame

    def to_numpy(self, box_type="ground"):
        '''
        :param box_type: Decide how to represent the box
        '''
        if len(self) == 0:
            return np.empty((0, 8))

        def to_ground(box): # store only 3D box and label
            cls_value = box.tag_top.value
            arr = np.concatenate([box.position, box.dimension, [box.yaw, cls_value]])
            return arr
        return np.stack([to_ground(box) for box in self])

    def to_torch(self, box_type="ground"):
        import torch
        return torch.tensor(self.to_numpy(box_type=box_type))

    def __str__(self):
        return "<ObjectTarget3DArray with %d objects>" % len(self)

CameraMetadata = namedtuple('CameraMetadata', [
    'width', 'height',
    'distort_coeffs', # coefficients of camera distortion model, follow OpenCV format
    'intri_matrix' # original intrinsic matrix used for cv2.undistortPoints
])
LidarMetadata = namedtuple('LidarMetadata', [])
RadarMetadata = namedtuple('RadarMetadata', [])
class TransformSet:
    '''
    This object load a collection of intrinsic and extrinsic parameters
    All extrinsic parameters are stored as transform from base frame to its frame
    In this class, we require all frames to use FLU coordinate including camera frame
    '''
    def __init__(self, base_frame):
        '''
        :param base_frame: name of base frame used by extrinsics
        '''
        self.base_frame = base_frame
        self.intrinsics = {} # projection matrics (mainly for camera)
        self.intrinsics_meta = {} # sensor metadata
        self.extrinsics = {} # transforms from base frame
        
    def _is_base(self, frame):
        return frame is None or frame == self.base_frame

    def _is_same(self, frame1, frame2):
        return (frame1 == frame2) or (self._is_base(frame1) and self._is_base(frame2))

    def _assert_exist(self, frame_id, extrinsic=False):
        if self._is_base(frame_id):
            return

        if frame_id not in self.intrinsics:
            raise ValueError("Frame {0} not found in intrinsic parameters, "
                "please add intrinsics for {0} first!".format(frame_id))

        if extrinsic and frame_id not in self.extrinsics:
            raise ValueError("Frame {0} not found in extrinsic parameters, "
                "please add extrinsic for {0} first!".format(frame_id))

    def set_intrinsic_general(self, frame_id, metadata=None):
        '''
        Set intrinsic for a general sensor.
        This is used for marking existence of a frame
        '''
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = metadata

    def set_intrinsic_camera(self, frame_id, transform, size, rotate=True, distort_coeffs=[], intri_matrix=None):
        '''
        Set camera intrinsics
        :param size: (width, height)
        :param rotate: if True, then transform will append an axis rotation (Front-Left-Up to Right-Down-Front)
        '''
        width, height = size
        if rotate:
            transform = transform.dot(np.array([
                [0,-1,0],
                [0,0,-1],
                [1,0,0]
            ]))

        self.intrinsics[frame_id] = transform
        self.intrinsics_meta[frame_id] = CameraMetadata(width, height, distort_coeffs, intri_matrix)

    def set_intrinsic_lidar(self, frame_id):
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = LidarMetadata()

    def set_intrinsic_radar(self, frame_id):
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = RadarMetadata()

    def set_intrinsic_pinhole(self, frame_id, size, cx, cy, fx, fy, s=0, distort_coeffs=[]):
        '''
        Set camera intrinsics with pinhole model parameters
        :param s: skew coefficient
        '''
        P = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
        self.set_intrinsic_camera(frame_id, P, size,
            rotate=True, distort_coeffs=distort_coeffs, intri_matrix=P)

    def set_extrinsic(self, transform, frame_to=None, frame_from=None):
        '''
        All extrinsics are stored as transform convert point from `frame_from` to `frame_to`
        :param frame_from: If set to None, then the source frame is base frame
        :param frame_to: If set to None, then the target frame is base frame
        '''
        if self._is_same(frame_to, frame_from): # including the case when frame_to=frame_from=None
            # the projection matrix need to be indentity
            assert np.allclose(np.diag(transform) == 1)
            assert np.sum(transform) == np.sum(np.diag(transform))

        if transform.shape == (3, 4):
            transform = np.vstack([transform, np.array([0]*3 + [1])])
        elif transform.shape != (4, 4):
            raise ValueError("Invalid matrix shape for extrinsics!")

        if self._is_base(frame_to):
            self._assert_exist(frame_from)
            self.extrinsics[frame_from] = np.linalg.inv(transform)
            return
        else:
            self._assert_exist(frame_to)

        if self._is_base(frame_from):
            self._assert_exist(frame_to)
            self.extrinsics[frame_to] = transform
            return
        else:
            self._assert_exist(frame_from)

        if frame_from in self.extrinsics and frame_to in self.extrinsics:
            raise ValueError("Frame %s and %s are both registered in extrinsic, "
                "please update one of them at one time" % (frame_to, frame_to))
        if frame_from in self.extrinsics:
            self.extrinsics[frame_to] = np.dot(transform, self.extrinsics[frame_from])
        elif frame_to in self.extrinsics:
            self.extrinsics[frame_from] = np.dot(transform, np.linalg.inv(self.extrinsics[frame_to]))
        else:
            raise ValueError("All frames are not present in extrinsics! "
                "Please add one of them first!")

    def get_extrinsic(self, frame_to=None, frame_from=None):
        '''
        :param frame_from: If set to None, then the source frame is base frame
        '''
        if self._is_same(frame_to, frame_from):
            return 1 # identity

        if not self._is_base(frame_from):
            self._assert_exist(frame_from, extrinsic=True)
            if not self._is_base(frame_to):
                self._assert_exist(frame_to, extrinsic=True)
                return np.dot(self.extrinsics[frame_to], np.linalg.inv(self.extrinsics[frame_from]))
            else:
                return np.linalg.inv(self.extrinsics[frame_from])
        else:
            if not self._is_base(frame_to):
                self._assert_exist(frame_to, extrinsic=True)
                return self.extrinsics[frame_to]
            else:
                return 1 # identity

    @property
    def frames(self):
        return list(self.intrinsics.keys())

    def transform_objects(self, objects: ObjectTarget3DArray, frame_to=None):
        '''
        Change the representing frame of a object array
        '''
        if self._is_same(objects.frame, frame_to):
            return

        rt = self.get_extrinsic(frame_from=objects.frame, frame_to=frame_to)
        r, t = Rotation.from_matrix(rt[:3, :3]), rt[:3, 3]
        new_objs = ObjectTarget3DArray(frame=frame_to)
        for obj in objects:
            position = np.dot(r.as_matrix(), obj.position) + t
            orientation = r * obj.orientation
            new_obj = ObjectTarget3D(position, orientation, obj.dimension, obj.tag, obj.id)
            new_objs.append(new_obj)
        return new_objs

    def project_points_to_camera(self, points, frame_to, frame_from=None, remove_outlier=True, return_dmask=False):
        '''
        :param remove_outlier: If set to True, only points that fall into image view will be returned
        :param return_dmask: also return the mask for z > 0 only
        :return: return points, mask and dmask if required. The masks are array of indices
        '''
        self._assert_exist(frame_from)
        self._assert_exist(frame_to)

        width, height, distorts, intri_matrix = self.intrinsics_meta[frame_to]
        rt = self.get_extrinsic(frame_to=frame_to, frame_from=frame_from)
        homo_xyz = np.insert(points[:, :3], 3, 1, axis=1)

        homo_uv = self.intrinsics[frame_to].dot(rt.dot(homo_xyz.T)[:3])
        d = homo_uv[2, :]
        u, v = homo_uv[0, :] / d, homo_uv[1, :] / d

        # mask points that are in camera view
        dmask = d > 0
        mask = (0 < u) & (u < width) & (0 < v) & (v < height) & dmask

        distorts = np.array(distorts)
        if distorts.size > 0:
            # save old mask with tolerance
            tolerance = 20
            mask = (-tolerance < u) & (u < width + tolerance) & (-tolerance < v) & (v < height + tolerance)

            # do distortion
            fx, fy, cx, cy = intri_matrix[0,0], intri_matrix[1,1], intri_matrix[0,2], intri_matrix[1,2]
            k1, k2, p1, p2, k3 = distorts
            u, v = (u - cx) / fx, (v - cy) / fy
            r2 = u*u + v*v
            auv, au, av = 2*u*v, r2 + 2*u*u, r2 + 2*v*v
            cdist = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
            icdist2 = 1 # 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6)
            ud0 = u*cdist*icdist2 + p1*auv + p2*au # + k[8]*r2 + k[9]*r4
            vd0 = v*cdist*icdist2 + p1*av + p2*auv # + k[10]*r2 + k[11]*r4
            u, v = ud0 * fx + cx, vd0 * fy + cy

            # mask again
            nmask = (0 < u) & (u < width) & (0 < v) & (v < height)
            mask = mask & nmask & dmask

        # filter points and return mask
        if remove_outlier:
            u, v = u[mask], v[mask]
        mask, = np.where(mask)
        dmask, = np.where(dmask)

        if return_dmask:
            return np.array([u, v]).T, mask, dmask
        else:
            return np.array([u, v]).T, mask
