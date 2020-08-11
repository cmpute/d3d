#pragma once

#include <torch/extension.h>
#include "d3d/geometry.hpp"

template <typename scalar_t, typename TBox>
struct _BoxUtilCuda
{
    static CUDA_CALLABLE_MEMBER TBox make_box(const _CudaSubAccessor(1) data);
    static CUDA_CALLABLE_MEMBER TBox make_box(__restrict__ scalar_t* data);
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, d3d::Box2f>
{
    static CUDA_CALLABLE_MEMBER d3d::Box2f make_box(const _CudaSubAccessor(1) data)
    {
        return d3d::poly2_from_xywhr<float>(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER d3d::Box2f make_box(__restrict__ scalar_t* data)
    {
        return d3d::poly2_from_xywhr<float>(data[0], data[1], data[2], data[3], data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, d3d::AABox2f>
{
    static CUDA_CALLABLE_MEMBER d3d::AABox2f make_box(const _CudaSubAccessor(1) data)
    {
        return d3d::aabox2_from_poly2(d3d::poly2_from_xywhr<float>(data[0], data[1], data[2], data[3], data[4]));
    }
    static CUDA_CALLABLE_MEMBER d3d::AABox2f make_box(__restrict__ scalar_t* data)
    {
        return d3d::aabox2_from_poly2(d3d::poly2_from_xywhr<float>(data[0], data[1], data[2], data[3], data[4]));
    }
};
