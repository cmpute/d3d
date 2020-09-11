from libcpp.vector cimport vector
cimport numpy as np

cdef class ObjectTag:
    cdef public object mapping # enum type
    cdef public vector[int] labels
    cdef public vector[float] scores

cdef class ObjectTarget3D:
    # variables with underscore at the end are cython variable, python version is exposed as property
    cdef float[:] position_, dimension_
    cdef float[:, :] position_var_, dimension_var_
    cdef public object orientation # FIXME: export scipy definition or directly store quaternion
    cdef public float orientation_var  # XXX: how to describe angle variance?
    cdef public unsigned long long tid # tid = 0 means no id assigned, so valid tid should be greater than 1
    cdef public ObjectTag tag

    cpdef np.ndarray to_numpy(self, str box_type=*)

cdef class TrackingTarget3D(ObjectTarget3D):
    cdef float[:] velocity_, angular_velocity_
    cdef float[:, :] velocity_var_, angular_velocity_var_
    cdef public float history

cdef class Target3DArray(list):
    cdef public str frame
    cdef public unsigned long long timestamp # unix timestamp in miliseconds

    cdef ObjectTarget3D get(self, int index)
    cdef TrackingTarget3D tget(self, int index)
    cpdef np.ndarray to_numpy(self, str box_type=*)

cdef class TransformSet:
    cdef public str base_frame
    cdef public dict intrinsics
    cdef public dict intrinsics_meta
    cdef public dict extrinsics

    cdef bint _is_base(self, str frame)
    cdef bint _is_same(self, str frame1, str frame2)
    cdef void _assert_exist(self, str frame_id, bint extrinsic=*)

    cpdef void set_intrinsic_general(self, str frame_id, object metadata=*)
    cpdef void set_intrinsic_camera(self, str frame_id, np.ndarray transform, size, rotate=*, distort_coeffs=*, intri_matrix=*)
    cpdef void set_intrinsic_lidar(self, str frame_id)
    cpdef void set_intrinsic_radar(self, str frame_id)
    cpdef void set_intrinsic_pinhole(self, str frame_id, size, cx, cy, fx, fy, s=*, distort_coeffs=*)
    cpdef void set_intrinsic_map_pin(self, str frame_id, lon=*, lat=*)
    cpdef void set_extrinsic(self, transform, str frame_to=*, str frame_from=*)
    cpdef np.ndarray get_extrinsic(self, str frame_to=*, str frame_from=*)

    cpdef Target3DArray transform_objects(self, Target3DArray objects, str frame_to=*)
    cpdef tuple project_points_to_camera(self, points, str frame_to, str frame_from=*, remove_outlier=*, return_dmask=*)
