#pragma once

#include <torch/extension.h>
#include "d3d/box/geometry.hpp"

template <typename scalar_t, typename TBox>
struct _BoxUtilCpu
{
    static TBox make_box(const _CpuAccessor(1) data);
};
template <typename scalar_t>
struct _BoxUtilCpu<scalar_t, Box2f>
{
    static Box2f make_box(const _CpuAccessor(1) data)
    {
        return Box2f(data[0], data[1], data[2], data[3], data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtilCpu<scalar_t, AABox2f>
{
    static AABox2f make_box(const _CpuAccessor(1) data)
    {
        return Box2f(data[0], data[1], data[2], data[3], data[4]).bbox();
    }
};

py::list rbox_2d_crop(torch::Tensor cloud, torch::Tensor boxes);
