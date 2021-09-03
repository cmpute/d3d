from libcpp cimport bool

cdef extern from "d3d/dgal_wrap.h" nogil:
    bool box3dr_contains(float x, float y, float z, float lx, float ly, float lz, float rz, float xq, float yq, float zq)
    float box3dr_pdist(float x, float y, float z, float lx, float ly, float lz, float rz, float xq, float yq, float zq)
    float box3dr_iou(float x1, float y1, float z1, float lx1, float ly1, float lz1, float rz1,
                     float x2, float y2, float z2, float lx2, float ly2, float lz2, float rz2)
    float box3d_iou(float x1, float y1, float z1, float lx1, float ly1, float lz1, float rz1,
                    float x2, float y2, float z2, float lx2, float ly2, float lz2, float rz2)
