#include "d3d/math/math.h"
#include "d3d/math/bessel.h"

using namespace torch;

template <typename scalar_t>
void i0e_templated(const _CpuAccessor(1) src_, _CpuAccessor(1) dst_)
{
    parallel_for(0, src_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        for (int i = begin; i < end; ++i)
            dst_[i] = i0e<scalar_t>(src_[i]);
    });
}

Tensor i0e(const Tensor& self)
{
    Tensor result = torch::empty_like(self, self.options());
    Tensor self_flat = self.flatten();
    Tensor result_flat = result.flatten();
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i0e", [&]{
        i0e_templated(self_flat._cpu_accessor(1), result_flat._cpu_accessor(1));
    });
    return result;
}

template <typename scalar_t>
void i1e_templated(const _CpuAccessor(1) src_, _CpuAccessor(1) dst_)
{
    parallel_for(0, src_.size(0), 0, [&](int64_t begin, int64_t end)
    {
        for (int i = begin; i < end; ++i)
            dst_[i] = i1e<scalar_t>(src_[i]);
    });
}

Tensor i1e(const Tensor& self)
{
    Tensor result = torch::empty_like(self, self.options());
    Tensor self_flat = self.flatten();
    Tensor result_flat = result.flatten();
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "i1e", [&]{
        i1e_templated(self_flat._cpu_accessor(1), result_flat._cpu_accessor(1));
    });
    return result;
}
