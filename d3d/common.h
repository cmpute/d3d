#ifndef D3D_COMMON_CUH
#define D3D_COMMON_CUH

// CUDA function descriptors
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_RESTRICT __restrict__
#define CUDA_CONSTANT __constant__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_RESTRICT
#define CUDA_CONSTANT
#endif 

// abbreviated pytorch accessors
#define _CpuAccessor(n) torch::TensorAccessor<scalar_t, n>
#define _CpuAccessorT(t,n) torch::TensorAccessor<t, n>
#define _cpu_accessor(n) accessor<scalar_t, n>()
#define _cpu_accessor_t(t,n) accessor<t, n>()

#define _CudaAccessor(n) torch::PackedTensorAccessor32<scalar_t,n,torch::RestrictPtrTraits>
#define _CudaAccessorT(t,n) torch::PackedTensorAccessor32<t,n,torch::RestrictPtrTraits>
#define _cuda_accessor(n) packed_accessor32<scalar_t,n,torch::RestrictPtrTraits>()
#define _cuda_accessor_t(t,n) packed_accessor32<t,n,torch::RestrictPtrTraits>()
#define _CudaSubAccessor(n) torch::TensorAccessor<scalar_t,n,torch::RestrictPtrTraits,int32_t>
#define _CudaSubAccessorT(t,n) torch::TensorAccessor<t,n,torch::RestrictPtrTraits,int32_t>

#ifdef NDEBUG
#define THREADS_COUNT 512 // XXX: 1024 is too large in some cases. try to increase this when the code is optimized
#else
#define THREADS_COUNT 16 // the code is less optimized, so the kernel could take larger memory
#endif

#define CUDA_CHECK_ERROR_SYNC(errorMessage) {                                \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaDeviceSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#endif // D3D_COMMON_CUH
