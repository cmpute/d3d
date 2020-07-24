
from libc.math cimport sqrt as sqrtf

cdef inline float wmean(float a, int wa, float b, int wb) nogil:
    # weighted mean
    if wa == 0: return b
    elif wb == 0: return a
    else: return (a * wa + b * wb) / (wa + wb)

cdef inline float diffnorm3(float[:] a, float[:] b) nogil:
    # calculate normed difference between two 3d vectors
    cdef float d0 = a[0] - b[0]
    cdef float d1 = a[1] - b[1]
    cdef float d2 = a[2] - b[2]
    return sqrtf(d0*d0 + d1*d1 + d2*d2)
