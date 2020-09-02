#pragma once

#include <torch/extension.h>

// define dispatch macros
#define _SCATTER_DISPATCH_ALIGNTYPE_CASE(ALIGNTYPE, ...) \
    case ALIGNTYPE:{                            \
        const AlignType Atype = ALIGNTYPE;      \
        return __VA_ARGS__();}

#define _SCATTER_DISPATCH_ALIGNTYPE(atype, ...)             \
    [&] { switch (atype)                                    \
    {                                                       \
    _SCATTER_DISPATCH_ALIGNTYPE_CASE(AlignType::MEAN, __VA_ARGS__)   \
    _SCATTER_DISPATCH_ALIGNTYPE_CASE(AlignType::LINEAR, __VA_ARGS__) \
    case AlignType::DROP:                                   \
    default:                                                \
        throw py::value_error("Unsupported align type!");   \
    }}

#define _SCATTER_DISPATCH_DIM_CASE(DIM, ...) \
    case DIM:{                      \
        const int Dim = DIM;        \
        return __VA_ARGS__();}

#define _SCATTER_DISPATCH_DIM(dim, ...)                         \
    [&] { switch (dim)                                          \
    {                                                           \
    _SCATTER_DISPATCH_DIM_CASE(1, __VA_ARGS__)                  \
    _SCATTER_DISPATCH_DIM_CASE(2, __VA_ARGS__)                  \
    _SCATTER_DISPATCH_DIM_CASE(3, __VA_ARGS__)                  \
    default:                                                    \
        throw py::value_error("Unsupported dimension size: " + to_string(dim)); \
    }}

// define function definitions
enum class AlignType : int { DROP=0, MEAN=1, LINEAR=2, MAX=3, NEAREST=4 };

torch::Tensor aligned_scatter_forward(
    const torch::Tensor coord, const torch::Tensor image_feature, const AlignType atype
);
void aligned_scatter_backward(
    const torch::Tensor coord, const torch::Tensor grad, const AlignType atype,
    torch::Tensor image_grad
);

#ifdef BUILD_WITH_CUDA
torch::Tensor aligned_scatter_forward_cuda(
    const torch::Tensor coord, const torch::Tensor image_feature, const AlignType atype
);
void aligned_scatter_backward_cuda(
    const torch::Tensor coord, const torch::Tensor grad, const AlignType atype,
    torch::Tensor image_grad
);
#endif
