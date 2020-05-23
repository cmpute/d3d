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

template <int Dim, AlignType AType>
void aligned_scatter_forward_templated(
    const Tensor coord, const Tensor image_feature, Tensor scatter_feature
)
{
    constexpr int Neighbor = 1 << Dim;
    auto coord_ = coord.accessor<float, 2>();
    auto image_feature_ = image_feature.accessor<float, Dim+2>();
    auto scatter_feature_ = scatter_feature.accessor<float, 2>();
    auto shape = coord.sizes();

    parallel_for(0, coord_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        for (int i = begin; i < end; ++i)
        {
            int b = coord_[i][0]; // batch index
            int lcoord[Neighbor][Dim]; // store neighbor coordinates
            float lweights[Neighbor]; // store neighbor weights
            if (AType == AlignType::LINEAR)
                for (int d = 0; d < Neighbor; d++)
                    lweights[d] = 1;

            // prepare coordinates and weights
            // here j is a bit mask for rounding direction, right most bit corresponds to coord[0]
            for (int j = 0; j < Neighbor; j++)
                for (int d = 0; d < Dim; d++)
                {
                    const int dmin = 0;
                    const int dmax = image_feature_.size(d+2) - 1;
                    const float dcoord = coord_[i][d+1];

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


Tensor aligned_scatter_forward(
    const Tensor coord, const Tensor image_feature, const AlignType atype
)
{
    Tensor scatter_feature = torch::empty({coord.size(0), image_feature.size(1)}, image_feature.options());
    _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
        aligned_scatter_forward_templated<Dim, Atype>(coord, image_feature, scatter_feature);
    }));
    return scatter_feature;
}
