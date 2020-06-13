#include "d3d/math/math.h"
#include "d3d/math/bessel.h"

using namespace torch;

template <typename scalar_t>
__global__ void i0e_kernel(const _CudaAccessor(1) src_, _CudaAccessor(1) dst_)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < src_.size(0))
        dst_[i] = i0e<scalar_t>(src_[i]);
}

Tensor i0e_cuda(const Tensor& self)
{
    Tensor result = torch::empty_like(self, self.options());
    Tensor self_flat = self.flatten();
    Tensor result_flat = result.flatten();

    const int threads = THREADS_COUNT;
    const int blocks = divup(self_flat.size(0), threads);

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i0e_cuda", [&]{
        i0e_kernel<scalar_t><<<blocks, threads>>>(
            self_flat._cuda_accessor(1), result_flat._cuda_accessor(1));
    });
    return result;
}

template <typename scalar_t>
__global__ void i1e_kernel(const _CudaAccessor(1) src_, _CudaAccessor(1) dst_)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < src_.size(0))
        dst_[i] = i1e<scalar_t>(src_[i]);
}

Tensor i1e_cuda(const Tensor& self)
{
    Tensor result = torch::empty_like(self, self.options());
    Tensor self_flat = self.flatten();
    Tensor result_flat = result.flatten();

    const int threads = THREADS_COUNT;
    const int blocks = divup(self_flat.size(0), threads);

    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i1e_cuda", [&]{
        i1e_kernel<scalar_t><<<blocks, threads>>>(
            self_flat._cuda_accessor(1), result_flat._cuda_accessor(1));
    });
    return result;
}
