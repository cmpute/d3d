#include "d3d/common.h"
#include "d3d/box/iou.h"
#include "d3d/box/utils.cuh"

using namespace std;
using namespace torch;
using namespace d3d;

template <typename scalar_t>
__global__ void iou2d_forward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_
) {
    using BoxType = AABox2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);
        ious_[i][j] = iou(bi, bj);
    }
}

Tensor iou2d_forward_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)},
        torch::dtype(boxes1.dtype()).device(boxes1.device()));
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2d_forward_cuda", [&] {
        iou2d_forward_kernel<scalar_t><<<blocks, threads>>>(
        boxes1._cuda_accessor(2),
        boxes2._cuda_accessor(2),
        ious._cuda_accessor(2));
    });

    return ious;
}

template <typename scalar_t>
__global__ void iou2d_backward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    const _CudaAccessor(2) grad_,
    _CudaAccessor(2) grad_boxes1_,
    _CudaAccessor(2) grad_boxes2_
) {
    using BoxType = AABox2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);
        BoxType grad_i, grad_j;
        iou_grad(bi, bj, grad_[i][j], grad_i, grad_j);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
    }
}

tuple<Tensor, Tensor> iou2d_backward_cuda(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);

    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "iou2d_backward_cuda", [&] {
        iou2d_backward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            grad._cuda_accessor(2),
            grad_boxes1._cuda_accessor(2),
            grad_boxes2._cuda_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}

template <typename scalar_t>
__global__ void iou2dr_forward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_,
    _CudaAccessorT(uint8_t, 2) nx_,
    _CudaAccessorT(uint8_t, 3) xflags_
) {
    using BoxType = Box2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);
        uint8_t flags[8];
        ious_[i][j] = iou(bi, bj, nx_[i][j], flags);

        #pragma unroll
        for (int k = 0; k < 8; k++)
            xflags_[i][j][k] = flags[k];
    }
}

tuple<Tensor, Tensor, Tensor> iou2dr_forward_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    Tensor nx = torch::empty({boxes1.size(0), boxes2.size(0)},
        torch::dtype(torch::kByte).device(boxes1.device()));
    Tensor xflags = torch::empty({boxes1.size(0), boxes2.size(0), 8},
        torch::dtype(torch::kByte).device(boxes1.device()));
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2dr_forward_cuda", [&] {
        iou2dr_forward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            ious._cuda_accessor(2),
            nx._cuda_accessor_t(uint8_t, 2),
            xflags._cuda_accessor_t(uint8_t, 3));
    });

    return make_tuple(ious, nx, xflags);
}

template <typename scalar_t>
__global__ void iou2dr_backward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    const _CudaAccessor(2) grad_,
    const _CudaAccessorT(uint8_t, 2) nx_,
    const _CudaAccessorT(uint8_t, 3) xflags_,
    _CudaAccessor(2) grad_boxes1_,
    _CudaAccessor(2) grad_boxes2_
) {
    using BoxType = Box2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);

        BoxType grad_i, grad_j;
        uint8_t flags[8];
        #pragma unroll
        for(int k = 0; k < 8; k++)
            flags[k] = xflags_[i][j][k];
    
        iou_grad(bi, bj, grad_[i][j], nx_[i][j], flags, grad_i, grad_j);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
    }
}

tuple<Tensor, Tensor> iou2dr_backward_cuda(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    const Tensor nx, const Tensor xflags
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "iou2dr_backward_cuda", [&] {
        iou2dr_backward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            grad._cuda_accessor(2),
            nx._cuda_accessor_t(uint8_t, 2),
            xflags._cuda_accessor_t(uint8_t, 3),
            grad_boxes1._cuda_accessor(2),
            grad_boxes2._cuda_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}


template <typename scalar_t>
__global__ void giou2dr_forward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_,
    _CudaAccessorT(uint8_t, 3) nxm_,
    _CudaAccessorT(uint8_t, 3) xmflags_
) {
    using BoxType = Box2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);
        uint8_t flags[16];
        ious_[i][j] = giou(bi, bj, nxm_[i][j][0], nxm_[i][j][1], flags, flags + 8);

        #pragma unroll
        for (int k = 0; k < 16; k++)
            xmflags_[i][j][k] = flags[k];
    }
}

tuple<Tensor, Tensor, Tensor> giou2dr_forward_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    Tensor nxm = torch::empty({boxes1.size(0), boxes2.size(0), 2},
        torch::dtype(torch::kByte).device(boxes1.device()));
    Tensor xmflags = torch::empty({boxes1.size(0), boxes2.size(0), 16},
        torch::dtype(torch::kByte).device(boxes1.device()));
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "giou2dr_forward_cuda", [&] {
        giou2dr_forward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            ious._cuda_accessor(2),
            nxm._cuda_accessor_t(uint8_t, 3),
            xmflags._cuda_accessor_t(uint8_t, 3));
    });

    return make_tuple(ious, nxm, xmflags);
}

template <typename scalar_t>
__global__ void giou2dr_backward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    const _CudaAccessor(2) grad_,
    const _CudaAccessorT(uint8_t, 3) nxm_,
    const _CudaAccessorT(uint8_t, 3) xmflags_,
    _CudaAccessor(2) grad_boxes1_,
    _CudaAccessor(2) grad_boxes2_
) {
    using BoxType = Box2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);

        BoxType grad_i, grad_j;
        uint8_t flags[16];
        #pragma unroll
        for(int k = 0; k < 16; k++)
            flags[k] = xmflags_[i][j][k];
    
        giou_grad(bi, bj, grad_[i][j], nxm_[i][j][0], nxm_[i][j][1], flags, flags + 8, grad_i, grad_j);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
    }
}

tuple<Tensor, Tensor> giou2dr_backward_cuda(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    const Tensor nxm, const Tensor xmflags
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "giou2dr_backward_cuda", [&] {
        giou2dr_backward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            grad._cuda_accessor(2),
            nxm._cuda_accessor_t(uint8_t, 3),
            xmflags._cuda_accessor_t(uint8_t, 3),
            grad_boxes1._cuda_accessor(2),
            grad_boxes2._cuda_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}


template <typename scalar_t>
__global__ void diou2dr_forward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_,
    _CudaAccessorT(uint8_t, 3) nxd_,
    _CudaAccessorT(uint8_t, 3) xflags_
) {
    using BoxType = Box2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);
        uint8_t flags[8];
        ious_[i][j] = diou(bi, bj, nxd_[i][j][0], nxd_[i][j][1], nxd_[i][j][2], flags);

        #pragma unroll
        for (int k = 0; k < 8; k++)
        xflags_[i][j][k] = flags[k];
    }
}

tuple<Tensor, Tensor, Tensor> diou2dr_forward_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    Tensor nxd = torch::empty({boxes1.size(0), boxes2.size(0), 3},
        torch::dtype(torch::kByte).device(boxes1.device()));
    Tensor xflags = torch::empty({boxes1.size(0), boxes2.size(0), 8},
        torch::dtype(torch::kByte).device(boxes1.device()));
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "diou2dr_forward_cuda", [&] {
        diou2dr_forward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            ious._cuda_accessor(2),
            nxd._cuda_accessor_t(uint8_t, 3),
            xflags._cuda_accessor_t(uint8_t, 3));
    });

    return make_tuple(ious, nxd, xflags);
}

template <typename scalar_t>
__global__ void diou2dr_backward_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    const _CudaAccessor(2) grad_,
    const _CudaAccessorT(uint8_t, 3) nxd_,
    const _CudaAccessorT(uint8_t, 3) xflags_,
    _CudaAccessor(2) grad_boxes1_,
    _CudaAccessor(2) grad_boxes2_
) {
    using BoxType = Box2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);

        BoxType grad_i, grad_j;
        uint8_t flags[8];
        #pragma unroll
        for(int k = 0; k < 8; k++)
            flags[k] = xflags_[i][j][k];
    
        diou_grad(bi, bj, grad_[i][j], nxd_[i][j][0], nxd_[i][j][1], nxd_[i][j][2], flags, grad_i, grad_j);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
    }
}

tuple<Tensor, Tensor> diou2dr_backward_cuda(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    const Tensor nxd, const Tensor xflags
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "diou2dr_backward_cuda", [&] {
        diou2dr_backward_kernel<scalar_t><<<blocks, threads>>>(
            boxes1._cuda_accessor(2),
            boxes2._cuda_accessor(2),
            grad._cuda_accessor(2),
            nxd._cuda_accessor_t(uint8_t, 3),
            xflags._cuda_accessor_t(uint8_t, 3),
            grad_boxes1._cuda_accessor(2),
            grad_boxes2._cuda_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}
