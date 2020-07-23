cimport numpy as np

cdef class ObjectTag:
    cdef public object mapping
    cdef public list labels
    cdef public list scores

cdef class ObjectTarget3D:
    cdef public float [:] position, dimension
    cdef public float [:, :] position_var, dimension_var
    cdef public float orientation_var  # XXX: how to describe angle variance?
    cdef public object orientation # TODO: export scipy definition
    cdef public object id
    cdef public ObjectTag tag

    cpdef np.ndarray to_numpy(self, str box_type=*)

cdef class ObjectTarget3DArray(list):
    cdef public str frame

    cpdef np.ndarray to_numpy(self, str box_type=*)
