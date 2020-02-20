#ifndef D3D_COMMON_CUH
#define D3D_COMMON_CUH

// CUDA function descriptors
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

// abbreviated pytorch accessors
#define _PackedAccessor(n) torch::PackedTensorAccessor32<scalar_t,n,torch::RestrictPtrTraits>
#define _PackedAccessorT(t,n) torch::PackedTensorAccessor32<t,n,torch::RestrictPtrTraits>
#define _packed_accessor(n) packed_accessor32<scalar_t,n,torch::RestrictPtrTraits>()
#define _packed_accessor_typed(t,n) packed_accessor32<t,n,torch::RestrictPtrTraits>()

#define DivUp(m,n) (((m)+(n)-1) / (n))

#endif // D3D_COMMON_CUH
