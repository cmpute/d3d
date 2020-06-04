#include <torch/extension.h>

enum class IouType : int { NA=0, BOX=1, RBOX=2 };
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


torch::Tensor nms2d(
    const torch::Tensor boxes, const torch::Tensor scores,
    const IouType iou_type, const SupressionType supression_type,
    const float iou_threshold, const float score_threshold, const float supression_param
);

torch::Tensor nms2d_cuda(
    const torch::Tensor boxes, const torch::Tensor scores,
    const IouType iou_type, const SupressionType supression_type,
    const float iou_threshold, const float score_threshold, const float supression_param
);
