#pragma once

#include <torch/extension.h>
#include "d3d/geometry.hpp"

template <typename scalar_t, typename TBox>
struct _BoxUtilCpu
{
    static TBox make_box(const _CpuAccessor(1) data);
};
template <typename scalar_t>
struct _BoxUtilCpu<scalar_t, d3d::Box2f>
{
    static d3d::Box2f make_box(const _CpuAccessor(1) data)
    {
        return d3d::poly2_from_xywhr<float>(data[0], data[1], data[2], data[3], data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtilCpu<scalar_t, d3d::AABox2f>
{
    static d3d::AABox2f make_box(const _CpuAccessor(1) data)
    {
        return d3d::aabox2_from_poly2(d3d::poly2_from_xywhr<float>(data[0], data[1], data[2], data[3], data[4]));
    }
};

py::list rbox_2d_crop(torch::Tensor cloud, torch::Tensor boxes);
