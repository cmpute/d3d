from libcpp.vector cimport vector
cimport numpy as np

cdef class ObjectTag:
    cdef public object mapping
    cdef public vector[int] labels
    cdef public vector[float] scores

cdef class ObjectTarget3D:
    # variables with underscore at the end are cython variable, python version is exposed as property
    cdef float[:] position_, dimension_
    cdef float[:, :] position_var_, dimension_var_
    cdef public object orientation # FIXME: export scipy definition
    cdef public float orientation_var  # XXX: how to describe angle variance?
    cdef public object tid # FIXME: convert all id (such as hash string) to integer
    cdef public ObjectTag tag

    cpdef np.ndarray to_numpy(self, str box_type=*)

cdef class ObjectTarget3DArray(list):
    cdef public str frame
    cdef public float timestamp

    cdef ObjectTarget3D get(self, int index)
    cpdef np.ndarray to_numpy(self, str box_type=*)

cdef class TrackingTarget3D(ObjectTarget3D):
    cdef float[:] velocity_, angular_velocity_
    cdef float[:, :] velocity_var_, angular_velocity_var_
    cdef public float history
