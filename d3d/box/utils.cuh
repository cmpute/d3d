#pragma once

#include <torch/extension.h>
#include "d3d/box/geometry.hpp"

template <typename scalar_t, typename TBox>
struct _BoxUtilCuda
{
    static CUDA_CALLABLE_MEMBER TBox make_box(const _CudaSubAccessor(1) data);
    static CUDA_CALLABLE_MEMBER TBox make_box(__restrict__ scalar_t* data);
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, Box2f>
{
    static CUDA_CALLABLE_MEMBER Box2f make_box(const _CudaSubAccessor(1) data)
    {
        return Box2f(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER Box2f make_box(__restrict__ scalar_t* data)
    {
        return Box2f(data[0], data[1], data[2], data[3], data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, AABox2f>
{
    static CUDA_CALLABLE_MEMBER AABox2f make_box(const _CudaSubAccessor(1) data)
    {
        return Box2f(data[0], data[1], data[2], data[3], data[4]).bbox();
    }
    static CUDA_CALLABLE_MEMBER AABox2f make_box(__restrict__ scalar_t* data)
    {
        return Box2f(data[0], data[1], data[2], data[3], data[4]).bbox();
    }
};
