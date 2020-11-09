#pragma once

#include <torch/extension.h>
#include "dgal/geometry.hpp"
#include "dgal/geometry_grad.hpp"

template <typename scalar_t, typename TBox>
struct _BoxUtilCuda
{
    static CUDA_CALLABLE_MEMBER TBox make_box(const _CudaSubAccessor(1) data);
    static CUDA_CALLABLE_MEMBER TBox make_box(__restrict__ scalar_t* data);
    static CUDA_CALLABLE_MEMBER void make_box_grad(const _CudaSubAccessor(1) data, const TBox &grad,
        _CudaSubAccessor(1) grad_data);
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, dgal::Quad2<scalar_t>>
{
    static CUDA_CALLABLE_MEMBER dgal::Quad2<scalar_t> make_box(const _CudaSubAccessor(1) data)
    {
        return dgal::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER dgal::Quad2<scalar_t> make_box(const __restrict__ scalar_t* data)
    {
        return dgal::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER void make_box_grad(const _CudaSubAccessor(1) data,
        const dgal::Quad2<scalar_t> &grad, _CudaSubAccessor(1) grad_data)
    {
        dgal::poly2_from_xywhr_grad(data[0], data[1], data[2], data[3], data[4], grad,
            grad_data[0], grad_data[1], grad_data[2], grad_data[3], grad_data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtilCuda<scalar_t, dgal::AABox2<scalar_t>>
{
    static CUDA_CALLABLE_MEMBER dgal::AABox2<scalar_t> make_box(const _CudaSubAccessor(1) data)
    {
        return dgal::aabox2_from_poly2(dgal::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]));
    }
    static CUDA_CALLABLE_MEMBER dgal::AABox2<scalar_t> make_box(const __restrict__ scalar_t* data)
    {
        return dgal::aabox2_from_poly2(dgal::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]));
    }
    static CUDA_CALLABLE_MEMBER void make_box_grad(const _CudaSubAccessor(1) data,
        const dgal::AABox2<scalar_t> &grad, _CudaSubAccessor(1) grad_data)
    {
        dgal::Quad2<scalar_t> poly, grad_poly;
        grad_poly.zero();
        poly = dgal::poly2_from_xywhr(data[0], data[1], data[2], data[3], data[4]);
        dgal::aabox2_from_poly2_grad(poly, grad, grad_poly);
        dgal::poly2_from_xywhr_grad(data[0], data[1], data[2], data[3], data[4], grad_poly,
            grad_data[0], grad_data[1], grad_data[2], grad_data[3], grad_data[4]);
    }
};
