#include <limits>
#include "d3d/common.h"
#include "d3d/point/scatter.h"

using namespace std;
using namespace torch;

// helper class to access tensor to various dimensions
template <typename scalar_t, int N, int I=0> inline
typename std::enable_if<(I==N-1), scalar_t&>::type
_var_access(_CpuAccessor(1) accessor, int* coord)
{
    return accessor[coord[I]];
}
template <typename scalar_t, int N, int I=0> inline
typename std::enable_if<(I<N-1), scalar_t&>::type
_var_access(_CpuAccessor(N-I) accessor, int* coord)
{
    return _var_access<scalar_t, N, I+1>(accessor[coord[I]], coord);
}

template<typename scalar_t> inline int _floor(scalar_t value)
{
    int i = int(value);
    if (i > value) return i-1; 
    else return i;
}
template<typename scalar_t> inline int _ceil(scalar_t value)
{
    int i = int(value);
    if (i < value) return i+1; 
    else return i;
}
template <typename scalar_t, int Dim, AlignType AType> inline void _fill_lcoords(
    const c10::IntArrayRef& image_shape,
    const _CpuAccessor(1) coord_,
    int lcoord[][Dim], scalar_t lweights[])
{
    constexpr int Neighbor = 1 << Dim;
    if (AType == AlignType::LINEAR)
        for (int d = 0; d < Neighbor; d++)
            lweights[d] = 1;

    // here j is a bit mask for rounding direction, right most bit corresponds to coord[0]
    for (int j = 0; j < Neighbor; j++)
        for (int d = 0; d < Dim; d++)
        {
            const int dmin = 0;
            const int dmax = image_shape.at(d+2) - 1;
            const scalar_t dcoord = coord_[d+1];

            if (dcoord > dmax)
            {
                lcoord[j][d] = dmax;
                if (AType == AlignType::LINEAR)
                    lweights[j] *= 0.5;
            }
            else if (dcoord < dmin)
            {
                lcoord[j][d] = dmin;
                if (AType == AlignType::LINEAR)
                    lweights[j] *= 0.5;
            }
            else if (j & (1u<<d))
            {
                lcoord[j][d] = _ceil(dcoord);
                if (AType == AlignType::LINEAR)
                    lweights[j] *= 1 + dcoord - lcoord[j][d];
            }
            else // j & (1u<<d) == 0
            {
                lcoord[j][d] = _floor(dcoord);
                if (AType == AlignType::LINEAR)
                    lweights[j] *= 1 - dcoord + lcoord[j][d];
            }
        }
}

template <typename scalar_t, int Dim, AlignType AType>
void aligned_scatter_forward_templated(
    const _CpuAccessor(2) coord_,
    const _CpuAccessor(Dim+2) image_feature_,
    _CpuAccessor(2) scatter_feature_
)
{
    parallel_for(0, coord_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        constexpr int Neighbor = 1 << Dim;
        for (int i = begin; i < end; ++i)
        {
            int b = coord_[i][0]; // batch index
            int lcoord[Neighbor][Dim]; // store neighbor coordinates
            scalar_t lweights[Neighbor]; // store neighbor weights

            // prepare coordinates and weights
            _fill_lcoords<scalar_t, Dim, AType>(image_feature_.sizes(), coord_[i], lcoord, lweights);

            // project features
            for (int c = 0; c < image_feature_.size(1); c++)
            {
                switch (AType)
                {
                    case AlignType::MEAN:
                    {
                        scalar_t sum = 0;
                        for (int j = 0; j < Neighbor; j++)
                            sum += _var_access<scalar_t, Dim>(image_feature_[b][c], lcoord[j]);
                        scatter_feature_[i][c] = sum / Neighbor;
                        break;
                    }
                    case AlignType::MAX:
                    {
                        scalar_t vmax = -numeric_limits<scalar_t>::lowest();
                        for (int j = 0; j < Neighbor; j++)
                            vmax = max(vmax, _var_access<scalar_t, Dim>(image_feature_[b][c], lcoord[j]));
                        scatter_feature_[i][c] = vmax;
                        break;
                    }
                    case AlignType::LINEAR:
                    {
                        scalar_t sum = 0;
                        for (int j = 0; j < Neighbor; j++)
                            sum += _var_access<scalar_t, Dim>(image_feature_[b][c], lcoord[j]) * lweights[j];
                        scatter_feature_[i][c] = sum;
                        break;
                    }
                    case AlignType::DROP:
                    default:
                        break;
                }
            }
        }
    });
}

template <typename scalar_t, int Dim, AlignType AType>
void aligned_scatter_backward_templated(
    const _CpuAccessor(2) coord_, 
    const _CpuAccessor(2) grad_,
    _CpuAccessor(Dim+2) image_grad_
) {
    parallel_for(0, coord_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        constexpr int Neighbor = 1 << Dim;
        for (int i = begin; i < end; ++i)
        {
            int b = coord_[i][0]; // batch index
            int lcoord[Neighbor][Dim]; // store neighbor coordinates
            scalar_t lweights[Neighbor]; // store neighbor weights
            _fill_lcoords<scalar_t, Dim, AType>(image_grad_.sizes(), coord_[i], lcoord, lweights);

            // project features
            for (int c = 0; c < image_grad_.size(1); c++)
            {
                switch (AType)
                {
                    case AlignType::MEAN:
                        for (int j = 0; j < Neighbor; j++)
                            _var_access<scalar_t, Dim>(image_grad_[b][c], lcoord[j]) += grad_[i][c] / Neighbor;
                        break;
                    case AlignType::LINEAR:
                        for (int j = 0; j < Neighbor; j++)
                            _var_access<scalar_t, Dim>(image_grad_[b][c], lcoord[j]) += grad_[i][c] * lweights[j];
                        break;
                    case AlignType::DROP:
                    default:
                        break;
                }
            }
        }
    });
}


Tensor aligned_scatter_forward(
    const Tensor coord, const Tensor image_feature, const AlignType atype
) {
    Tensor scatter_feature = torch::empty({coord.size(0), image_feature.size(1)}, image_feature.options());
    AT_DISPATCH_FLOATING_TYPES(image_feature.scalar_type(), "aligned_scatter_forward",
        _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
            aligned_scatter_forward_templated<scalar_t, Dim, Atype>(
                coord._cpu_accessor(2),
                image_feature._cpu_accessor(Dim+2),
                scatter_feature._cpu_accessor(2));
        }))
    );
    return scatter_feature;
}

void aligned_scatter_backward(
    const Tensor coord, const Tensor grad, const AlignType atype, torch::Tensor image_grad
) {
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "aligned_scatter_backward", ([&] {
        _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
            aligned_scatter_backward_templated<scalar_t, Dim, Atype>(
                coord._cpu_accessor(2),
                grad._cpu_accessor(2),
                image_grad._cpu_accessor(Dim+2));
        }));
    }));
}