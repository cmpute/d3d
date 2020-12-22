# cython: language_level=3, embedsignature=True

import base64
import enum
import logging
import pickle
from collections import namedtuple
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.spatial.transform import Rotation
from cpython.list cimport PyList_GetItem, PyList_Size

_logger = logging.getLogger("d3d")

def _d3d_enum_mapping():
    import d3d.dataset as dd
    return {
        # 0 for non-built-in mapping
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
    def __init__(self, labels, mapping=None, scores=None):
        if mapping is not None and not issubclass(mapping, enum.Enum):
            raise ValueError("The object class mapping should be an Enum")
        self.mapping = mapping

        # sanity check
        if scores is None:
            if isinstance(labels, (list, tuple)) and len(labels) != 1:
                raise ValueError("There cannot be multiple labels without scores")
            labels = [labels]
            scores = [1]
        else:
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            if not isinstance(scores, (list, tuple)):
                scores = [scores]

        # convert labels to enum id value
        for i in range(len(labels)):
            if isinstance(labels[i], str):
                labels[i] = self.mapping[labels[i]].value
            elif isinstance(labels[i], int):
                continue
            else:
                if self.mapping is None: # infer mapping type
                    self.mapping = type(labels[i])
                labels[i] = labels[i].value
                
        # sort labels descending
        order = list(reversed(np.argsort(scores)))
        self.labels = [labels[i] for i in order]
        self.scores = [scores[i] for i in order]

    def __str__(self):
        name = self.mapping(self.labels[0]).name
        return "<ObjectTag, top class: %s>" % name

    def serialize(self):
        '''
        Serialize this object to primitives
        '''
        return (_d3d_enum_mapping()[self.mapping], self.labels, self.scores)

    @staticmethod
    def deserialize(data):
        '''
        Deserialize this object to primitives
        '''
        mapping = _d3d_enum_lookup()[data[0]]
        return ObjectTag(data[1], mapping, data[2])

cdef inline float[:] create_vector3(values):
    if len(values) != 3:
        raise ValueError("Incorrect vector length")
    return np.asarray(values, dtype=np.float32)

cdef inline float[:, :] create_matrix33(values):
    if values is None:
        return np.zeros((3, 3), dtype=np.float32)
    else:
        return np.asarray(values, dtype=np.float32).reshape(3, 3)

cdef inline bytes pack_ull(unsigned long long value):
    cdef list result = []
    while value > 0:
        result.append(value % 256)
        value = value // 256
    return bytes(result)

cdef class ObjectTarget3D:
    '''
    This class stands for a target in cartesian coordinate. The body coordinate is FLU (front-left-up).
    '''
    def __init__(self, position, orientation, dimension, tag,
        tid=0, position_var=None, orientation_var=None, dimension_var=None):
        '''
        :param position: Position of object center (x,y,z)
        :param orientation: Object heading (direction of x-axis attached on body)
            with regard to x-axis of the world at the object center.
        :param dimension: Length of the object in 3 dimensions (lx,ly,lz)
        :param tag: Classification information of the object
        :param tid: ID of the object used for tracking (optional), 0 means no tracking id assigned
        '''

        self.position_ = create_vector3(position)
        self.dimension_ = create_vector3(dimension)

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

        self.tid = tid
        self.position_var_ = create_matrix33(position_var)
        self.dimension_var_ = create_matrix33(dimension_var)
        self.orientation_var = 0 if orientation_var is None else orientation_var

    # exposes basic members
    @property
    def position(self):
        return np.asarray(self.position_)
    @position.setter
    def position(self, value):
        self.position_ = create_vector3(value)
    @property
    def position_var(self):
        return np.asarray(self.position_var_)
    @position_var.setter
    def position_var(self, value):
        self.position_var_ = create_matrix33(value)
    @property
    def dimension(self):
        return np.asarray(self.dimension_)
    @dimension.setter
    def dimension(self, value):
        self.dimension_ = create_vector3(value)
    @property
    def dimension_var(self):
        return np.asarray(self.dimension_var_)
    @dimension_var.setter
    def dimension_var(self, value):
        self.dimension_var_ = create_matrix33(value)

    @property
    def tag_top(self):
        return self.tag.mapping(self.tag.labels[0])

    @property
    def tag_name(self):
        '''
        Return the name of the target's top tag
        '''
        return self.tag_top.name

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
        if abs(angles[1]) + abs(angles[2]) > 0.2:
            message = "The roll (%.2f) and pitch(%.2f) angle in objects may be to large to ignore!" % \
                (angles[2], angles[1])
            _logger.warning(message)
            warn(message)
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

    @property
    def tid64(self):
        '''
        Return base64 represented tracking id
        '''
        return base64.b64encode(pack_ull(self.tid)).rstrip(b'=').decode()

    cpdef np.ndarray to_numpy(self, str box_type="ground"):
        cls_value = self.tag_top.value
        cdef np.ndarray arr = np.concatenate([self.position, self.dimension,
            [self.yaw, cls_value, self.tag_score]])
        return arr

    def serialize(self):
        return (
            np.asarray(self.position_).tolist(),
            np.ravel(self.position_var_).tolist(),
            np.asarray(self.dimension_).tolist(),
            np.ravel(self.dimension_var_).tolist(),
            self.orientation.as_quat().tolist(),
            self.orientation_var,
            self.tid,
            self.tag.serialize()
        )

    @classmethod
    def deserialize(cls, data):
        pos, pos_var, dim, dim_var, ori_data, ori_var, tid, tag_data = data
        ori = Rotation.from_quat(ori_data)
        tag = ObjectTag.deserialize(tag_data)
        return cls(pos, ori, dim, tag, tid=tid,
            position_var=pos_var, orientation_var=ori_var, dimension_var=dim_var
        )

cdef class TrackingTarget3D(ObjectTarget3D):
    def __init__(self, position, orientation, dimension, velocity, angular_velocity, tag,
        tid=0, position_var=None, orientation_var=None, dimension_var=None,
        velocity_var=None, angular_velocity_var=None, history=None):

        self.position_ = create_vector3(position)
        self.dimension_ = create_vector3(dimension)
        self.velocity_ = create_vector3(velocity)
        self.angular_velocity_ = create_vector3(angular_velocity)

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

        self.tid = tid
        self.history = history or float('nan')

        self.position_var_ = create_matrix33(position_var)
        self.dimension_var_ = create_matrix33(dimension_var)
        self.orientation_var = 0 if orientation_var is None else orientation_var
        self.velocity_var_ = create_matrix33(velocity_var)
        self.angular_velocity_var_ = create_matrix33(angular_velocity_var)

    # exposes basic members
    @property
    def velocity(self):
        return np.asarray(self.velocity_)
    @velocity.setter
    def velocity(self, value):
        self.velocity_ = create_vector3(value)
    @property
    def velocity_var(self):
        return np.asarray(self.velocity_var_)
    @velocity_var.setter
    def velocity_var(self, value):
        self.velocity_var_ = create_matrix33(value)
    @property
    def angular_velocity(self):
        return np.asarray(self.angular_velocity_)
    @angular_velocity.setter
    def angular_velocity(self, value):
        self.angular_velocity_ = create_vector3(value)
    @property
    def angular_velocity_var(self):
        return np.asarray(self.angular_velocity_var_)
    @angular_velocity_var.setter
    def angular_velocity_var(self, value):
        self.angular_velocity_var_ = create_matrix33(value)

    def serialize(self):
        return (
            np.asarray(self.position_).tolist(),
            np.ravel(self.position_var_).tolist(),
            np.asarray(self.dimension_).tolist(),
            np.ravel(self.dimension_var_).tolist(),
            self.orientation.as_quat().tolist(),
            self.orientation_var,
            np.asarray(self.velocity_).tolist(),
            np.asarray(self.velocity_var_).tolist(),
            np.asarray(self.angular_velocity_).tolist(),
            np.asarray(self.angular_velocity_var_).tolist(),
            self.tid,
            self.tag.serialize(),
            self.history
        )

    @classmethod
    def deserialize(cls, data):
        (pos, pos_var, dim, dim_var, ori_data, ori_var,
            vel, vel_var, avel, avel_var, tid, tag_data, history) = data
        ori = Rotation.from_quat(ori_data)
        tag = ObjectTag.deserialize(tag_data)
        return cls(pos, ori, dim, vel, avel, tag, tid=tid,
            position_var=pos_var, orientation_var=ori_var, dimension_var=dim_var,
            velocity_var=vel_var, angular_velocity_var=avel_var, history=history
        )

    cpdef np.ndarray to_numpy(self, str box_type="ground"):
        cls_value = self.tag_top.value
        cdef np.ndarray arr = np.concatenate([self.position, self.dimension,
            [self.yaw, cls_value, self.tag_score]])
        return arr

cdef class Target3DArray(list):
    def __init__(self, iterable=[], frame=None, timestamp=0):
        '''
        :param frame: Frame that the box parameters used. None means base frame (in TransformSet)
        '''
        super().__init__(iterable)
        self.frame = frame
        self.timestamp = timestamp

        # copy frame value
        if isinstance(iterable, Target3DArray) and not frame:
            self.frame = iterable.frame
            self.timestamp = iterable.timestamp

    cdef ObjectTarget3D get(self, int index):
        return <ObjectTarget3D>(PyList_GetItem(self, index))

    cdef TrackingTarget3D tget(self, int index):
        return <TrackingTarget3D>(PyList_GetItem(self, index))

    cdef Py_ssize_t size(self):
        return PyList_Size(self)

    cpdef np.ndarray to_numpy(self, str box_type="ground"):
        '''
        :param box_type: Decide how to represent the box. {ground: box projected along z axis}
        '''
        if len(self) == 0:
            return np.empty((0, 9))
        return np.stack([box.to_numpy(box_type) for box in self])

    def to_torch(self, box_type="ground"):
        import torch
        return torch.from_numpy(self.to_numpy(box_type))

    def serialize(self):
        if len(self) > 0:
            if any(type(obj) != type(self[0]) for obj in self):
                raise ValueError("All elements are required to be the same type (ObjectTarget3D"
                    "or TrackingTarget3D) before dumping!")
            type_mapping = {
                # empty list is 0
                ObjectTarget3D: 1,
                TrackingTarget3D: 2
            }
            type_code = type_mapping[type(self[0])]
        else:
            type_code = 0
        return (self.frame, self.timestamp, type_code, [obj.serialize() for obj in self])

    @staticmethod
    def deserialize(data):
        if data[2] == 1:
            objs = [ObjectTarget3D.deserialize(obj) for obj in data[3]]
        elif data[2] == 2:
            objs = [TrackingTarget3D.deserialize(obj) for obj in data[3]]
        else:
            assert data[2] == 0 and len(data[3]) == 0
            objs = []
        return Target3DArray(objs, frame=data[0], timestamp=data[1])

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
                return Target3DArray.deserialize(msgpack.unpackb(fout.read()))

    def __repr__(self):
        return "<Target3DArray with %d objects @ %s>" % (len(self), self.frame)

    def filter_tag(self, tags):
        '''
        Filter the list by select only objects with given tags

        :param tags: None means no filter, otherwise str/enum or list of str/enum
        '''
        if not tags:
            return self
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        tags = [str(t) if not isinstance(t, str) else t for t in tags] # use tag name to filter
        tags = [t.lower() for t in tags]
        return Target3DArray([box for box in self if box.tag_name.lower() in tags], self.frame)

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

    def homo(self):
        '''
        Convert the pose to a homogeneous matrix representation

        Note that the pose is represented by position and orientation. The rotation operation is
        performed after translation `R(x+T)`, which is different from homogeneous transform
        matrix`Rx+T`.
        '''
        arr = np.eye(4)
        arr[:3, :3] = self.orientation.as_matrix()
        arr[:3, 3] = self.orientation.as_matrix().dot(self.position)
        return arr

    def __repr__(self):
        return "<EgoPose %s>" % str(self)

    def __str__(self):
        ypr = tuple(self.orientation.as_euler("ZYX").tolist())
        return "position: [x=%.2f, y=%.2f, z=%.2f], orientation: [r=%.2f, p=%.2f, y=%.2f]" % \
            (tuple(self.position.tolist()) + ypr[::-1])

cdef class CameraMetadata:
    def __init__(self, int width, int height, np.ndarray distort_coeffs, np.ndarray intri_matrix, float mirror_coeff):
        self.width = width
        self.height = height
        self.distort_coeffs = distort_coeffs
        self.intri_matrix = intri_matrix
        self.mirror_coeff = mirror_coeff

cdef class LidarMetadata:
    def __init__(self):
        pass

cdef class RadarMetadata:
    def __init__(self):
        pass

cdef class PinMetadata:
    def __init__(self, float lon, float lat):
        self.lon = lon
        self.lat = lat

cdef class TransformSet:
    '''
    This object load a collection of intrinsic and extrinsic parameters
    All extrinsic parameters are stored as transform from base frame to its frame
    In this class, we require all frames to use FLU coordinate including camera frame
    '''
    def __init__(self, str base_frame):
        '''
        :param base_frame: name of base frame used by extrinsics
        '''
        self.base_frame = base_frame
        self.intrinsics = {} # projection matrics (mainly for camera)
        self.intrinsics_meta = {} # sensor metadata
        self.extrinsics = {} # transforms from base frame
        
    cdef bint _is_base(self, str frame):
        return frame is None or frame == self.base_frame

    cdef bint _is_same(self, str frame1, str frame2):
        return (frame1 == frame2) or (self._is_base(frame1) and self._is_base(frame2))

    cdef void _assert_exist(self, str frame_id, bint extrinsic=False):
        if self._is_base(frame_id):
            return

        if frame_id not in self.intrinsics:
            raise ValueError("Frame {0} not found in intrinsic parameters, "
                "please add intrinsics for {0} first!".format(frame_id))

        if extrinsic and frame_id not in self.extrinsics:
            raise ValueError("Frame {0} not found in extrinsic parameters, "
                "please add extrinsic for {0} first!".format(frame_id))

    cpdef void set_intrinsic_general(self, str frame_id, object metadata=None) except*:
        '''
        Set intrinsic for a general sensor.
        This is used for marking existence of a frame
        '''
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = metadata

    cpdef void set_intrinsic_camera(self, str frame_id, np.ndarray transform, size, bint rotate=True,
        np.ndarray distort_coeffs=None, np.ndarray intri_matrix=None, mirror_coeff=float('nan')) except*:
        '''
        Set camera intrinsics
        :param size: (width, height)
        :param rotate: if True, then transform will append an axis rotation (Front-Left-Up to Right-Down-Front)
        :param distort_coeffs: distortion coefficients, see [OpenCV](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html) for details
        :param intri_matrix: intrinsic matrix (in general camera model)
        :param mirror_coeff: the xi coefficient in MEI camera model. Reference: Single View Point OmnidirectionalCamera Calibration from Planar Grids
        '''
        width, height = size
        if rotate:
            transform = transform.dot(np.array([
                [0,-1,0],
                [0,0,-1],
                [1,0,0]
            ]))

        self.intrinsics[frame_id] = transform
        self.intrinsics_meta[frame_id] = CameraMetadata(width, height,
            distort_coeffs, intri_matrix, mirror_coeff)

    cpdef void set_intrinsic_lidar(self, str frame_id) except*:
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = LidarMetadata()

    cpdef void set_intrinsic_radar(self, str frame_id) except*:
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = RadarMetadata()

    cpdef void set_intrinsic_pinhole(self, str frame_id, size, cx, cy, fx, fy, s=0, distort_coeffs=[]) except*:
        '''
        Set camera intrinsics with pinhole model parameters
        :param s: skew coefficient
        '''
        P = np.array([[fx, s, cx], [0, fy, cy], [0, 0, 1]])
        self.set_intrinsic_camera(frame_id, P, size,
            rotate=True, distort_coeffs=np.asarray(distort_coeffs), intri_matrix=P)

    cpdef void set_intrinsic_map_pin(self, str frame_id, lon=float('nan'), lat=float('nan')) except*:
        self.intrinsics[frame_id] = None
        self.intrinsics_meta[frame_id] = PinMetadata(lon, lat)

    cpdef void set_extrinsic(self, transform, str frame_to=None, str frame_from=None) except*:
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

    cpdef np.ndarray get_extrinsic(self, str frame_to=None, str frame_from=None):
        '''
        :param frame_from: If set to None, then the source frame is base frame
        '''
        if self._is_same(frame_to, frame_from):
            return np.eye(4)

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
                return np.eye(4)

    @property
    def frames(self):
        '''
        Report registered frame names (excluding base_frame)
        '''
        return list(self.intrinsics.keys())
    def __repr__(self):
        return "<TransformSet with frames: *%s>" % ", ".join([self.base_frame] + self.frames)

    cpdef Target3DArray transform_objects(self, Target3DArray objects, str frame_to=None):
        '''
        Change the representing frame of a object array
        '''
        if self._is_same(objects.frame, frame_to):
            return objects

        rt = self.get_extrinsic(frame_from=objects.frame, frame_to=frame_to)
        r, t = Rotation.from_matrix(rt[:3, :3]), rt[:3, 3]
        new_objs = Target3DArray(frame=frame_to)
        for obj in objects:
            position = np.dot(r.as_matrix(), obj.position) + t
            orientation = r * obj.orientation
            # TODO: also rotate the velocity

            if isinstance(obj, TrackingTarget3D): # notice that TrackingTarget3D derives from ObjectTarget3D
                velocity = np.dot(r.as_matrix(), obj.velocity)
                new_objs.append(TrackingTarget3D(
                    position=position, position_var=obj.position_var,
                    orientation=orientation, orientation_var=obj.orientation_var,
                    dimension=obj.dimension, dimension_var=obj.dimension_var,
                    velocity=velocity, velocity_var=obj.velocity_var,
                    angular_velocity=obj.angular_velocity, angular_velocity_var=obj.angular_velocity_var,
                    tag=obj.tag, tid=obj.tid, history=obj.history
                ))
            elif isinstance(obj, ObjectTarget3D):
                new_objs.append(ObjectTarget3D(
                    position=position, position_var=obj.position_var,
                    orientation=orientation, orientation_var=obj.orientation_var,
                    dimension=obj.dimension, dimension_var=obj.dimension_var,
                    tag=obj.tag, tid=obj.tid
                ))
            else:
                raise ValueError("Unsupported target type!")
        return new_objs

    cpdef np.ndarray transform_points(self, np.ndarray points, str frame_to, str frame_from=None):
        '''
        Convert point cloud from `frame_from` to `frame_to`
        '''
        rt = self.get_extrinsic(frame_to, frame_from)
        xyz = points[:, :3].dot(rt[:3, :3].T) + rt[:3, 3]
        return np.concatenate((xyz, points[:, 3:]), axis=1)

    cpdef tuple project_points_to_camera(self, np.ndarray points, str frame_to, str frame_from=None, remove_outlier=True, return_dmask=False):
        '''
        :param remove_outlier: If set to True, the mask will be applied, i.e. only points
            that fall into image view will be returned
        :param return_dmask: also return the mask for z > 0 only
        :return: return points, mask and dmask if required. The masks are array of indices
        '''
        self._assert_exist(frame_from)
        self._assert_exist(frame_to)

        meta = self.intrinsics_meta[frame_to] 
        rt = self.get_extrinsic(frame_to=frame_to, frame_from=frame_from)
        homo_xyz = np.insert(points[:, :3], 3, 1, axis=1)

        homo_uv = self.intrinsics[frame_to].dot(rt.dot(homo_xyz.T)[:3])
        d = homo_uv[2, :]
        u, v = homo_uv[0, :] / d, homo_uv[1, :] / d

        # mask points that are in camera view
        dmask = d > 0
        mask = (0 < u) & (u < meta.width) & (0 < v) & (v < meta.height) & dmask

        distorts = np.array(meta.distort_coeffs or [])
        if distorts.size > 0:
            # save old mask with tolerance
            tolerance = 20
            mask = (-tolerance < u) & (u < meta.width + tolerance) &\
                   (-tolerance < v) & (v < meta.height + tolerance)

            # do distortion
            intri_matrix = meta.intri_matrix
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
            nmask = (0 < u) & (u < meta.width) & (0 < v) & (v < meta.height)
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
