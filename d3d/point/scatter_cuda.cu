#include "d3d/point/scatter.h"
#include "d3d/common.h"
#include <limits>

#include "d3d/point/atomics.cuh"

using namespace std;
using namespace torch;

// helper class to access tensor to various dimensions
template <typename scalar_t, int N, int I=0> __device__ inline
typename std::enable_if<(I==N-1), scalar_t&>::type
_var_access(TensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t> accessor, int* coord)
{
    return accessor[coord[I]];
}
template <typename scalar_t, int N, int I=0> __device__ inline
typename std::enable_if<(I<N-1), scalar_t&>::type
_var_access(TensorAccessor<scalar_t, N-I, RestrictPtrTraits, int32_t> accessor, int* coord)
{
    return _var_access<scalar_t, N, I+1>(accessor[coord[I]], coord);
}

template<typename scalar_t> __device__ inline int _floor(scalar_t value)
{
    int i = int(value);
    if (i > value) return i-1; 
    else return i;
}
template<typename scalar_t> __device__ inline int _ceil(scalar_t value)
{
    int i = int(value);
    if (i < value) return i+1; 
    else return i;
}

template <typename scalar_t, int Dim, AlignType AType>
__global__ void aligned_scatter_forward_kernel(
    const _PackedAccessor(2) coord_,
    const _PackedAccessor(Dim+2) image_feature_,
    _PackedAccessor(2) scatter_feature_
)
{
    constexpr int Neighbor = 1 << Dim;
    __shared__ int lcoord[Neighbor][Dim]; // store neighbor coordinates
    __shared__ scalar_t lweights[Neighbor]; // store neighbor weights

    const int i = blockIdx.y;
    const int c = blockIdx.x * THREADS_COUNT + threadIdx.x;
    const int nchannel_inblock = min(image_feature_.size(1) - blockIdx.x * THREADS_COUNT, THREADS_COUNT);

    // intialize weight
    if (threadIdx.x < Neighbor)
    {
        if (AType == AlignType::LINEAR)
            lweights[threadIdx.x] = 1;
    }
    __syncthreads();

    // prepare coordinates
    // here j is a bit mask for rounding direction, right most bit corresponds to coord[0]
    if (threadIdx.x < Neighbor * Dim)
    {
        int j = threadIdx.x / Dim;
        int d = threadIdx.x % Dim;
        
        const int dmin = 0;
        const int dmax = image_feature_.size(d+2) - 1;
        const scalar_t dcoord = coord_[i][d+1];

        if (dcoord > dmax)
        {
            lcoord[j][d] = dmax;
            if (AType == AlignType::LINEAR)
                atomMul(lweights + j, 0.5);
        }
        else if (dcoord < dmin)
        {
            lcoord[j][d] = dmin;
            if (AType == AlignType::LINEAR)
                atomMul(lweights + j, 0.5);
        }
        else if (j & (1u<<d))
        {
            lcoord[j][d] = _ceil(dcoord);
            if (AType == AlignType::LINEAR)
                atomMul(lweights + j, 1 + dcoord - lcoord[j][d]);
        }
        else // j & (1u<<d) == 0
        {
            lcoord[j][d] = _floor(dcoord);
            if (AType == AlignType::LINEAR)
                atomMul(lweights + j, 1 - dcoord + lcoord[j][d]);
        }
    }
    __syncthreads();

    // project features
    int b = coord_[i][0]; // batch index
    if (c < nchannel_inblock)
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

template <int Dim, AlignType AType> inline
void aligned_scatter_forward_templated(
    const Tensor coord, const Tensor image_feature, Tensor scatter_feature
) {
    const int npoints = coord.size(0);
    const int nchannels = image_feature.size(1);

    const int threads = THREADS_COUNT;
    const dim3 blocks(DivUp(nchannels, THREADS_COUNT), npoints);

    // constexpr AlignType _ATYPE = AType; // pass template argument
    AT_DISPATCH_FLOATING_TYPES(image_feature.type(), "aligned_scatter_forward_cuda", [&] {
        aligned_scatter_forward_kernel<scalar_t, Dim, AType><<<blocks, threads>>>(
            coord._packed_accessor(2),
            image_feature._packed_accessor(Dim+2),
            scatter_feature._packed_accessor(2));
    });
}

Tensor aligned_scatter_forward_cuda(
    const Tensor coord, const Tensor image_feature, const AlignType atype
)
{
    Tensor scatter_feature = torch::empty({coord.size(0), image_feature.size(1)}, image_feature.options());
    _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
        aligned_scatter_forward_templated<Dim, Atype>(coord, image_feature, scatter_feature);
    }));
    return scatter_feature;
}
