#pragma once

#include <torch/extension.h>
#include "d3d/geometry.hpp"
#include "d3d/geometry_grad.hpp"

template <typename scalar_t, typename TBox>
struct _BoxUtilCuda
{
    static CUDA_CALLABLE_MEMBER TBox make_box(const _CudaSubAccessor(1) data);
    static CUDA_CALLABLE_MEMBER TBox make_box(__restrict__ scalar_t* data);
    static CUDA_CALLABLE_MEMBER void make_box_grad(const _CudaSubAccessor(1) data, const TBox &grad,
        _CudaSubAccessor(1) grad_data);
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, d3d::Quad2<scalar_t>>
{
    static CUDA_CALLABLE_MEMBER d3d::Quad2<scalar_t> make_box(const _CudaSubAccessor(1) data)
    {
        return d3d::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER d3d::Quad2<scalar_t> make_box(const __restrict__ scalar_t* data)
    {
        return d3d::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER void make_box_grad(const _CudaSubAccessor(1) data,
        const d3d::Quad2<scalar_t> &grad, _CudaSubAccessor(1) grad_data)
    {
        d3d::poly2_from_xywhr_grad(data[0], data[1], data[2], data[3], data[4], grad,
            grad_data[0], grad_data[1], grad_data[2], grad_data[3], grad_data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, d3d::AABox2<scalar_t>>
{
    static CUDA_CALLABLE_MEMBER d3d::AABox2<scalar_t> make_box(const _CudaSubAccessor(1) data)
    {
        return d3d::aabox2_from_poly2(d3d::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]));
    }
    static CUDA_CALLABLE_MEMBER d3d::AABox2<scalar_t> make_box(const __restrict__ scalar_t* data)
    {
        return d3d::aabox2_from_poly2(d3d::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]));
    }
    static CUDA_CALLABLE_MEMBER void make_box_grad(const _CudaSubAccessor(1) data,
        const d3d::AABox2<scalar_t> &grad, _CudaSubAccessor(1) grad_data)
    {
        d3d::Quad2<scalar_t> poly, grad_poly;
        poly = d3d::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]);
        d3d::aabox2_from_poly2_grad(poly, grad, grad_poly);
        d3d::poly2_from_xywhr_grad(data[0], data[1], data[2], data[3], data[4], grad_poly,
            grad_data[0], grad_data[1], grad_data[2], grad_data[3], grad_data[4]);
    }
};
