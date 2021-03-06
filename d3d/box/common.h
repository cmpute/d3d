#pragma once

#include "d3d/common.h"

enum class IouType : int { NA=0,
    BOX=1, RBOX=2, // traditional iou for axis aligned box and rotated box
    GBOX=3, GRBOX=4, // Generalized iou
    DBOX=5, DRBOX=6 // Distance iou
};
enum class SupressionType : int { HARD=0, LINEAR=1, GAUSSIAN=2 };

// define dispatch macros
#define _NMS_DISPATCH_IOUTYPE_CASE(IOUTYPE, ...)    \
    case IOUTYPE:{                                  \
        const IouType Iou = IOUTYPE;              \
        return __VA_ARGS__();}

#define _NMS_DISPATCH_IOUTYPE(itype, ...)                   \
    [&] { switch (itype)                                    \
    {                                                       \
    _NMS_DISPATCH_IOUTYPE_CASE(IouType::BOX, __VA_ARGS__)   \
    _NMS_DISPATCH_IOUTYPE_CASE(IouType::RBOX, __VA_ARGS__)  \
    case IouType::NA:                                       \
    default:                                                \
        throw py::value_error("Unsupported iou type!");     \
    }}

#define _NMS_DISPATCH_SUPRESSTYPE_CASE(SUPRESSTYPE, ...)    \
    case SUPRESSTYPE:{                                      \
        const SupressionType Supression = SUPRESSTYPE;      \
        return __VA_ARGS__();}

#define _NMS_DISPATCH_SUPRESSTYPE(itype, ...)                   \
    [&] { switch (itype)                                        \
    {                                                           \
    _NMS_DISPATCH_SUPRESSTYPE_CASE(SupressionType::HARD, __VA_ARGS__)       \
    _NMS_DISPATCH_SUPRESSTYPE_CASE(SupressionType::LINEAR, __VA_ARGS__)     \
    _NMS_DISPATCH_SUPRESSTYPE_CASE(SupressionType::GAUSSIAN, __VA_ARGS__)   \
    default:                                                    \
        throw py::value_error("Unsupported supression type!");  \
    }}
