import enum
import pickle
import logging
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_logger = logging.getLogger("d3d")

def _d3d_enum_mapping():
    import d3d.dataset as dd
    return {
        dd.kitti.KittiObjectClass: 1,
        dd.waymo.WaymoObjectClass: 2,
        dd.nuscenes.NuscenesObjectClass: 3,
        dd.nuscenes.NuscenesDetectionClass: 4
    }

def _d3d_enum_lookup():
    return {v: k for k, v in _d3d_enum_mapping().items()}

cdef class ObjectTag:
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

    def serialize(self):
        '''
        Serialize this object to primitives
        '''
        labels = [l.value for l in self.labels]
        return (_d3d_enum_mapping()[self.mapping], labels, self.scores)

    @staticmethod
    def deserialize(data):
        '''
        Deserialize this object to primitives
        TODO: test this
        '''
        mapping = _d3d_enum_lookup()[data[0]]
        labels = [mapping(l) for l in data[1]]
        return ObjectTag(data[1], mapping, data[2])

cdef class ObjectTarget3D:
    '''
    This class stands for a target in cartesian coordinate. The body coordinate is FLU (front-left-up).
    '''
    def __init__(self, position, orientation, dimension, tag,
        id_=None, position_var=None, orientation_var=None, dimension_var=None):
        '''
        :param position: Position of object center (x,y,z)
        :param orientation: Object heading (direction of x-axis attached on body)
            with regard to x-axis of the world at the object center.
        :param dimension: Length of the object in 3 dimensions (lx,ly,lz)
        :param tag: Classification information of the object
        :param id: ID of the object used for tracking (optional)
        '''

        assert len(position) == 3, "Invalid position shape"
        self.position = np.asarray(position, dtype=np.float32)

        assert len(dimension) == 3, "Invalid dimension shape"
        self.dimension = np.asarray(dimension, dtype=np.float32)

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

        self.id = id_
        if position_var is None:
            self.position_var = np.zeros((3, 3), dtype=np.float32)
        else:
            self.position_var = np.asarray(position_var, dtype=np.float32).reshape(3, 3)
        if dimension_var is None:
            self.dimension_var = np.zeros((3, 3), dtype=np.float32)
        else:
            self.dimension_var = np.asarray(dimension_var, dtype=np.float32).reshape(3, 3)
        # self.orientation_var = orientation_var or 0 # XXX: how to describe angle variance?

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

    cpdef np.ndarray to_numpy(self, str box_type="ground"):
        # store only 3D box and label
        cls_value = self.tag_top.value
        cdef np.ndarray arr = np.concatenate([self.position, self.dimension, [self.yaw, cls_value]])
        return arr

    def serialize(self):
        return (
            np.asarray(self.position).tolist(),
            np.ravel(self.position_var).tolist(),
            np.asarray(self.dimension).tolist(),
            np.ravel(self.dimension_var).tolist(),
            self.orientation.as_quat().tolist(),
            self.id,
            self.tag.serialize()
        )

    @staticmethod
    def deserialize(data):
        pos, pos_var, dim, dim_var, ori_data, id_, tag_data = data
        ori = Rotation.from_quat(ori_data)
        tag = ObjectTag.deserialize(tag_data)
        return ObjectTarget3D(pos, ori, dim, tag, id_, pos_var, dim_var)

cdef class ObjectTarget3DArray(list):
    def __init__(self, iterable=[], frame=None):
        '''
        :param frame: Frame that the box parameters used. None means base frame (in TransformSet)
        '''
        super().__init__(iterable)
        self.frame = frame

        # copy frame value
        if isinstance(iterable, ObjectTarget3DArray) and not frame:
            self.frame = iterable.frame

    cpdef np.ndarray to_numpy(self, str box_type="ground"):
        '''
        :param box_type: Decide how to represent the box. {ground: box projected along z axis}
        '''
        if len(self) == 0:
            return np.empty((0, 8))
        return np.stack([box.to_ground(box_type) for box in self])

    def to_torch(self, box_type="ground"):
        import torch
        return torch.tensor(self.to_numpy(box_type))

    def serialize(self):
        return (self.frame, [obj.serialize() for obj in self])

    @staticmethod
    def deserialize(data):
        objs = [ObjectTarget3D.deserialize(obj) for obj in data[1]]
        return ObjectTarget3DArray(objs, frame=data[0])

    def dump(self, output):
        import msgpack
        if isinstance(output, (str, Path)):
            with Path(output).open('wb') as fout:
                fout.write(msgpack.packb(self.serialize(), use_single_float=True))

    @staticmethod
    def load(output):
        import msgpack
        if isinstance(output, (str, Path)):
            with Path(output).open('rb') as fout:
                return ObjectTarget3DArray.deserialize(msgpack.unpackb(fout.read()))

    def __repr__(self):
        return "<ObjectTarget3DArray with %d objects>" % len(self)

    def filter_tag(self, tags):
        '''
        Filter the list by select only objects with given tags

        :param tags: None means no filter, otherwise str/enum or list of str/enum
        '''
        if not tags:
            return self
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        tags = (str(t) if not isinstance(t, str) else t for t in tags) # use tag name to filter
        tags = [t.lower() for t in tags]
        return ObjectTarget3DArray([box for box in self if box.tag_name.lower() in tags], self.frame)

class EgoPose:
    '''
    This object is used to store dynamic state of ego vehicle. All value is represented
        in earth-fixed coordinate (absolute coordinate).
    '''
    def __init__(self, position, orientation, position_var=None, orientation_var=None):
        
        assert len(position) == 3, "Invalid position shape"
        self.position = np.array(position)

        if isinstance(orientation, Rotation):
            self.orientation = orientation
        elif len(orientation) == 4:
            self.orientation = Rotation.from_quat(orientation)
        else:
            raise ValueError("Invalid rotation format")

        self.position_var = np.zeros((3, 3)) if position_var is None else position_var
        self.orientation_var = np.zeros((3, 3)) if orientation_var is None else orientation_var
 
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
        '''
        Report registered frame names (excluding base_frame)
        '''
        return list(self.intrinsics.keys())
    def __repr__(self):
        return "<TransformSet with frames: *%s>" % ", ".join([self.base_frame] + self.frames)

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
        :param remove_outlier: If set to True, the mask will be applied, i.e. only points
            that fall into image view will be returned
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

    def dump(self, output):
        if isinstance(output, (str, Path)):
            with Path(output).open('wb') as fout:
                pickle.dump(self, fout)

    @staticmethod
    def load(output):
        if isinstance(output, (str, Path)):
            with Path(output).open('rb') as fout:
                return pickle.load(fout)
