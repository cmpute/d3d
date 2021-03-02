# cython: language_level=3, embedsignature=True

from libc.stdint cimport uint8_t, uint16_t
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np

cdef class ObjectTag:

    cdef public object mapping
    '''
    The Enum class defining the label classes
    '''

    cdef public vector[int] labels
    '''
    The ids of the labels, sorted by score in descending order
    '''

    cdef public vector[float] scores
    '''
    The ids of the labels, sorted by score in descending order
    '''

cdef class ObjectTarget3D:
    # variables with underscore at the end are cython variable, python version is exposed as property
    cdef float[:] position_, dimension_
    cdef float[:, :] position_var_, dimension_var_
    cdef float[:] orientation_

    cdef public float orientation_var  # XXX: how to describe angle variance?
    '''
    Variance of orientation of the target. This API may be changed in future
    '''

    cdef public unsigned long long tid
    '''
    The unique id of the target (across frames).
    tid = 0 means no id assigned, so valid tid should be greater than 1.
    '''

    cdef public ObjectTag tag
    '''
    The tag attached to the target
    '''

    cpdef np.ndarray to_numpy(self, str box_type=*)
    cdef _crop(self, const float[:,:] cloud, bool[:] result)

cdef class TrackingTarget3D(ObjectTarget3D):
    cdef float[:] velocity_, angular_velocity_
    cdef float[:, :] velocity_var_, angular_velocity_var_

    cdef public float history
    '''
    Tracked time of this target in seconds
    '''

cdef class Target3DArray(list):
    cdef public str frame
    '''
    The transform frame which the targets lie in
    '''

    cdef public unsigned long long timestamp
    '''
    The timestamp of when the targets are annotated or reported.
    It's represented by unix timestamp in milliseconds
    '''

    cdef ObjectTarget3D get(self, int index)
    cdef TrackingTarget3D tget(self, int index)
    cdef Py_ssize_t size(self)
    cpdef np.ndarray to_numpy(self, str box_type=*)
    cdef void _crop_points(self, const float[:,:] cloud, bool[:,:] result)
    cdef void _paint_id(self, const bool[:,:] mask, const uint8_t[:] semantics, uint16_t[:] idarr)

cdef class CameraMetadata:
    cdef public int width
    '''
    Width of the camera image
    '''

    cdef public int height
    '''
    Height of the camera image
    '''

    cdef public np.ndarray distort_coeffs
    '''
    Coefficients of camera distortion model, follow OpenCV format
    '''

    cdef public np.ndarray intri_matrix
    '''
    Original intrinsic matrix used for cv2.undistortPoints
    '''

    cdef public float mirror_coeff
    '''
    Coefficient of mirror equation (as used in MEI camera model)
    '''

cdef class LidarMetadata:
    pass

cdef class RadarMetadata:
    pass

cdef class PinMetadata:
    cdef public float lon
    '''
    Longitude coordinate of the pin
    '''

    cdef public float lat
    '''
    Latitude coordinate of the pin
    '''

cdef class EgoPose:
    cdef public np.ndarray position
    '''
    The position of the ego sensor
    '''

    cdef float[:] orientation_

    cdef public np.ndarray position_var
    '''
    Variance of the estimation of the sensor position
    '''

    cdef public np.ndarray orientation_var
    '''
    Variance of the estimation of the sensor orientation
    '''

cdef class TransformSet:
    cdef public str base_frame
    cdef public dict intrinsics
    cdef public dict intrinsics_meta
    cdef public dict extrinsics

    cdef bint _is_base(self, str frame)
    cdef bint _is_same(self, str frame1, str frame2)
    cdef void _assert_exist(self, str frame_id, bint extrinsic=*)

    cpdef void set_intrinsic_general(self, str frame_id, object metadata=*) except*
    cpdef void set_intrinsic_camera(self, str frame_id, np.ndarray transform,
        size, bint rotate=*, distort_coeffs=*, np.ndarray intri_matrix=*, float mirror_coeff=*) except*
    cpdef void set_intrinsic_lidar(self, str frame_id) except*
    cpdef void set_intrinsic_radar(self, str frame_id) except*
    cpdef void set_intrinsic_pinhole(self, str frame_id, size, float cx, float cy,
        float fx, float fy, float s=*, distort_coeffs=*) except*
    cpdef void set_intrinsic_map_pin(self, str frame_id, lon=*, lat=*) except*
    cpdef void set_extrinsic(self, transform, str frame_to=*, str frame_from=*) except*
    cpdef np.ndarray get_extrinsic(self, str frame_to=*, str frame_from=*)

    cpdef Target3DArray transform_objects(self, Target3DArray objects, str frame_to=*)
    cpdef np.ndarray transform_points(self, np.ndarray points, str frame_to, str frame_from=*)
    cpdef tuple project_points_to_camera(self, np.ndarray points, str frame_to, str frame_from=*, bint remove_outlier=*, bint return_dmask=*)
