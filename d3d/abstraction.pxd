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
    cdef public float orientation_var  # XXX: how to describe angle variance?
    cdef public object orientation # FIXME: export scipy definition
    cdef public object tid # FIXME: convert all id (such as hash string) to integer
    cdef public ObjectTag tag

    cpdef np.ndarray to_numpy(self, str box_type=*)

cdef class ObjectTarget3DArray(list):
    cdef public str frame

    cdef ObjectTarget3D get(self, int index)
    cpdef np.ndarray to_numpy(self, str box_type=*)
