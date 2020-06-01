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

template<typename scalar_t, int Dim, AlignType AType> __device__ inline
void _fill_lcoords(
    const _CudaAccessor(Dim+2)& image_feature_,
    const TensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>& coord_,
    int lcoord[][Dim], scalar_t lweights[]
) {
    constexpr int Neighbor = 1 << Dim;

    // intialize weight
    if (threadIdx.x < Neighbor)
    {
        if (AType == AlignType::LINEAR)
            lweights[threadIdx.x] = 1;
    }
    __syncthreads();

    // here j is a bit mask for rounding direction, right most bit corresponds to coord[0]
    if (threadIdx.x < Neighbor * Dim)
    {
        int j = threadIdx.x / Dim;
        int d = threadIdx.x % Dim;
        
        const int dmin = 0;
        const int dmax = image_feature_.size(d+2) - 1;
        const scalar_t dcoord = coord_[d+1];

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
}

template <typename scalar_t, int Dim, AlignType AType>
__global__ void aligned_scatter_forward_kernel(
    const _CudaAccessor(2) coord_,
    const _CudaAccessor(Dim+2) image_feature_,
    _CudaAccessor(2) scatter_feature_
) {
    constexpr int Neighbor = 1 << Dim;
    __shared__ int lcoord[Neighbor][Dim]; // store neighbor coordinates
    __shared__ scalar_t lweights[Neighbor]; // store neighbor weights

    const int i = blockIdx.x;
    const int c = blockIdx.y * THREADS_COUNT + threadIdx.x;
    const int nchannel_inblock = min(image_feature_.size(1) - blockIdx.y * THREADS_COUNT, THREADS_COUNT);

    // prepare coordinates
    _fill_lcoords<scalar_t, Dim, AType>(image_feature_, coord_[i], lcoord, lweights);

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

template <typename scalar_t, int Dim, AlignType AType>
__global__ void aligned_scatter_backward_kernel(
    const _CudaAccessor(2) coord_,
    const _CudaAccessor(2) grad_,
    _CudaAccessor(Dim+2) image_grad_
) {
    constexpr int Neighbor = 1 << Dim;
    __shared__ int lcoord[Neighbor][Dim]; // store neighbor coordinates
    __shared__ scalar_t lweights[Neighbor]; // store neighbor weights

    const int i = blockIdx.x;
    const int c = blockIdx.y * THREADS_COUNT + threadIdx.x;
    const int nchannel_inblock = min(grad_.size(1) - blockIdx.y * THREADS_COUNT, THREADS_COUNT);

    // prepare coordinates
    _fill_lcoords<scalar_t, Dim, AType>(image_grad_, coord_[i], lcoord, lweights);

    // project features
    int b = coord_[i][0]; // batch index
    if (c < nchannel_inblock)
    {
        switch (AType)
        {
            case AlignType::MEAN:
                for (int j = 0; j < Neighbor; j++)
                {
                    scalar_t* val = &_var_access<scalar_t, Dim>(image_grad_[b][c], lcoord[j]);
                    atomAdd(val, grad_[i][c] / Neighbor);
                }
                break;
            case AlignType::LINEAR:
                for (int j = 0; j < Neighbor; j++)
                {
                    scalar_t* val = &_var_access<scalar_t, Dim>(image_grad_[b][c], lcoord[j]);
                    atomAdd(val, grad_[i][c] * lweights[j]);
                }
                break;
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
    const dim3 blocks(npoints, divup(nchannels, THREADS_COUNT));

    AT_DISPATCH_FLOATING_TYPES(image_feature.scalar_type(), "aligned_scatter_forward_cuda", [&] {
        aligned_scatter_forward_kernel<scalar_t, Dim, AType><<<blocks, threads>>>(
            coord._cuda_accessor(2),
            image_feature._cuda_accessor(Dim+2),
            scatter_feature._cuda_accessor(2));
    });
}

template <int Dim, AlignType AType> inline
void aligned_scatter_backward_templated(
    const Tensor coord, const Tensor grad, Tensor image_grad
) {
    const int npoints = coord.size(0);
    const int nchannels = image_grad.size(1);

    const int threads = THREADS_COUNT;
    const dim3 blocks(npoints, divup(nchannels, THREADS_COUNT));

    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "aligned_scatter_backward_cuda", [&] {
        aligned_scatter_backward_kernel<scalar_t, Dim, AType><<<blocks, threads>>>(
            coord._cuda_accessor(2),
            grad._cuda_accessor(2),
            image_grad._cuda_accessor(Dim+2));
    });
}

Tensor aligned_scatter_forward_cuda(
    const Tensor coord, const Tensor image_feature, const AlignType atype
) {
    Tensor scatter_feature = torch::empty({coord.size(0), image_feature.size(1)}, image_feature.options());
    _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
        aligned_scatter_forward_templated<Dim, Atype>(coord, image_feature, scatter_feature);
    }));
    return scatter_feature;
}

void aligned_scatter_backward_cuda(
    const Tensor coord, const Tensor grad, const AlignType atype, torch::Tensor image_grad
) {
    _SCATTER_DISPATCH_DIM(coord.size(1) - 1, _SCATTER_DISPATCH_ALIGNTYPE(atype, [&] {
        aligned_scatter_backward_templated<Dim, Atype>(coord, grad, image_grad);
    }));
}
