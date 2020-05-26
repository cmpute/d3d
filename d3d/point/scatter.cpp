#include <limits>
#include "d3d/point/scatter.h"

using namespace std;
using namespace torch;

// helper class to access tensor to various dimensions
template <typename scalar_t, int N, int I=0> inline
typename std::enable_if<(I==N-1), scalar_t&>::type
_var_access(TensorAccessor<scalar_t, 1> accessor, int* coord)
{
    return accessor[coord[I]];
}
template <typename scalar_t, int N, int I=0> inline
typename std::enable_if<(I<N-1), scalar_t&>::type
_var_access(TensorAccessor<scalar_t, N-I> accessor, int* coord)
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
template <int Dim, AlignType AType> inline void _fill_lcoords(
    const c10::IntArrayRef& image_shape,
    const TensorAccessor<float, 1>& coord_,
    int lcoord[][Dim], float lweights[])
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
            const float dcoord = coord_[d+1];

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

template <int Dim, AlignType AType>
void aligned_scatter_forward_templated(
    const Tensor coord, const Tensor image_feature, Tensor scatter_feature
)
{
    auto coord_ = coord.accessor<float, 2>();
    auto image_feature_ = image_feature.accessor<float, Dim+2>();
    auto scatter_feature_ = scatter_feature.accessor<float, 2>();

    parallel_for(0, coord_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        constexpr int Neighbor = 1 << Dim;
        for (int i = begin; i < end; ++i)
        {
            int b = coord_[i][0]; // batch index
            int lcoord[Neighbor][Dim]; // store neighbor coordinates
            float lweights[Neighbor]; // store neighbor weights

            // prepare coordinates and weights
            _fill_lcoords<Dim, AType>(image_feature.sizes(), coord_[i], lcoord, lweights);

            // project features
            for (int c = 0; c < image_feature_.size(1); c++)
            {
                switch (AType)
                {
                    case AlignType::MEAN:
                    {
                        float sum = 0;
                        for (int j = 0; j < Neighbor; j++)
                            sum += _var_access<float, Dim>(image_feature_[b][c], lcoord[j]);
                        scatter_feature_[i][c] = sum / Neighbor;
                        break;
                    }
                    case AlignType::MAX:
                    {
                        float vmax = -numeric_limits<float>::lowest();
                        for (int j = 0; j < Neighbor; j++)
                            vmax = max(vmax, _var_access<float, Dim>(image_feature_[b][c], lcoord[j]));
                        scatter_feature_[i][c] = vmax;
                        break;
                    }
                    case AlignType::LINEAR:
                    {
                        float sum = 0;
                        for (int j = 0; j < Neighbor; j++)
                            sum += _var_access<float, Dim>(image_feature_[b][c], lcoord[j]) * lweights[j];
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

template <int Dim, AlignType AType>
void aligned_scatter_backward_templated(
    const Tensor coord, const Tensor grad, Tensor image_grad
)
{
    auto coord_ = coord.accessor<float, 2>();
    auto grad_ = grad.accessor<float, 2>();
    auto image_grad_ = image_grad.accessor<float, Dim+2>();

    parallel_for(0, coord_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        constexpr int Neighbor = 1 << Dim;
        for (int i = begin; i < end; ++i)
        {
            int b = coord_[i][0]; // batch index
            int lcoord[Neighbor][Dim]; // store neighbor coordinates
            float lweights[Neighbor]; // store neighbor weights
            _fill_lcoords<Dim, AType>(image_grad.sizes(), coord_[i], lcoord, lweights);

            // project features
            for (int c = 0; c < image_grad_.size(1); c++)
            {
                switch (AType)
                {
                    case AlignType::MEAN:
                        for (int j = 0; j < Neighbor; j++)
                            _var_access<float, Dim>(image_grad_[b][c], lcoord[j]) += grad_[i][c] / Neighbor;
                        break;
                    case AlignType::LINEAR:
                        for (int j = 0; j < Neighbor; j++)
                            _var_access<float, Dim>(image_grad_[b][c], lcoord[j]) += grad_[i][c] * lweights[j];
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
    _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
        aligned_scatter_forward_templated<Dim, Atype>(coord, image_feature, scatter_feature);
    }));
    return scatter_feature;
}

void aligned_scatter_backward(
    const Tensor coord, const Tensor grad, const AlignType atype, torch::Tensor image_grad
) {
    _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
        aligned_scatter_backward_templated<Dim, Atype>(coord, grad, image_grad);
    }));
}