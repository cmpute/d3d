
#include "d3d/box/iou.h"
#include "d3d/box/utils.h"
#include "d3d/geometry.hpp"
#include "d3d/geometry_grad.hpp"

using namespace std;
using namespace torch;
using namespace d3d;

template <typename scalar_t>
void iou2d_forward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_
) {
    using BoxType = AABox2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            ious_[i][j] = iou(bi, bj);
        }
    });
}

Tensor iou2d_forward(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2d_forward", [&] {
        iou2d_forward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2));
    });
    return ious;
}

template <typename scalar_t>
void iou2d_backward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    const _CpuAccessor(2) grad_,
    _CpuAccessor(2) grad_boxes1_,
    _CpuAccessor(2) grad_boxes2_
) {
    using BoxType = AABox2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            BoxType grad_i, grad_j;
            iou_grad(bi, bj, grad_[i][j], grad_i, grad_j);
             _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
             _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
        }
    });
}

tuple<Tensor, Tensor> iou2d_backward(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "iou2d_backward", [&] {
        iou2d_backward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            grad._cpu_accessor(2),
            grad_boxes1._cpu_accessor(2),
            grad_boxes2._cpu_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}

template <typename scalar_t>
void iou2dr_forward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_,
    _CpuAccessorT(uint8_t, 2) nx_,
    _CpuAccessorT(uint8_t, 3) xflags_
) {
    using BoxType = Quad2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            uint8_t flags[8];
            ious_[i][j] = iou(bi, bj, nx_[i][j], flags);

            #pragma unroll
            for (int k = 0; k < 8; k++)
                xflags_[i][j][k] = flags[k];
        }
    });
}

tuple<Tensor, Tensor, Tensor> iou2dr_forward(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    Tensor nx = torch::empty({boxes1.size(0), boxes2.size(0)}, torch::dtype(torch::kByte));
    Tensor xflags = torch::empty({boxes1.size(0), boxes2.size(0), 8}, torch::dtype(torch::kByte));

    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2dr_forward", [&] {
        iou2dr_forward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2),
            nx._cpu_accessor_t(uint8_t, 2),
            xflags._cpu_accessor_t(uint8_t, 3));
    });
    return make_tuple(ious, nx, xflags);
}

template <typename scalar_t>
void iou2dr_backward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    const _CpuAccessor(2) grad_,
    const _CpuAccessorT(uint8_t, 2) nx_,
    const _CpuAccessorT(uint8_t, 3) xflags_,
    _CpuAccessor(2) grad_boxes1_,
    _CpuAccessor(2) grad_boxes2_
) {
    using BoxType = Quad2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);

            BoxType grad_i, grad_j;
            grad_i.zero(); grad_j.zero();
            uint8_t flags[8];
            #pragma unroll
            for(int k = 0; k < 8; k++)
                flags[k] = xflags_[i][j][k];
        
            iou_grad(bi, bj, grad_[i][j], nx_[i][j], flags, grad_i, grad_j);
            _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
            _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
        }
    });
}


tuple<Tensor, Tensor> iou2dr_backward(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    const Tensor nx, const Tensor xflags
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "iou2dr_backward", [&] {
        iou2dr_backward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            grad._cpu_accessor(2),
            nx._cpu_accessor_t(uint8_t, 2),
            xflags._cpu_accessor_t(uint8_t, 3),
            grad_boxes1._cpu_accessor(2),
            grad_boxes2._cpu_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}

template <typename scalar_t>
void giou2dr_forward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_,
    _CpuAccessorT(uint8_t, 3) nxm_,
    _CpuAccessorT(uint8_t, 3) xmflags_
) {
    using BoxType = Quad2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            uint8_t flags[16];
            ious_[i][j] = giou(bi, bj, nxm_[i][j][0], nxm_[i][j][1], flags, flags + 8);

            #pragma unroll
            for (int k = 0; k < 16; k++)
                xmflags_[i][j][k] = flags[k];
        }
    });
}

tuple<Tensor, Tensor, Tensor> giou2dr_forward(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    Tensor nxm = torch::empty({boxes1.size(0), boxes2.size(0), 2}, torch::dtype(torch::kByte));
    Tensor xmflags = torch::empty({boxes1.size(0), boxes2.size(0), 16}, torch::dtype(torch::kByte));

    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "giou2dr_forward", [&] {
        giou2dr_forward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2),
            nxm._cpu_accessor_t(uint8_t, 3),
            xmflags._cpu_accessor_t(uint8_t, 3));
    });
    return make_tuple(ious, nxm, xmflags);
}

template <typename scalar_t>
void giou2dr_backward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    const _CpuAccessor(2) grad_,
    const _CpuAccessorT(uint8_t, 3) nxm_,
    const _CpuAccessorT(uint8_t, 3) xmflags_,
    _CpuAccessor(2) grad_boxes1_,
    _CpuAccessor(2) grad_boxes2_
) {
    using BoxType = Quad2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);

            BoxType grad_i, grad_j;
            grad_i.zero(); grad_j.zero();
            uint8_t flags[16];
            #pragma unroll
            for(int k = 0; k < 16; k++)
                flags[k] = xmflags_[i][j][k];
        
            giou_grad(bi, bj, grad_[i][j], nxm_[i][j][0], nxm_[i][j][1], flags, flags + 8, grad_i, grad_j);
            _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
            _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
        }
    });
}


tuple<Tensor, Tensor> giou2dr_backward(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    const Tensor nxm, const Tensor flags
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "giou2dr_backward", [&] {
        giou2dr_backward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            grad._cpu_accessor(2),
            nxm._cpu_accessor_t(uint8_t, 3),
            flags._cpu_accessor_t(uint8_t, 3),
            grad_boxes1._cpu_accessor(2),
            grad_boxes2._cpu_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}

template <typename scalar_t>
void diou2dr_forward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    _CpuAccessor(2) ious_,
    _CpuAccessorT(uint8_t, 3) nxd_,
    _CpuAccessorT(uint8_t, 3) xflags_
) {
    using BoxType = Quad2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);
            uint8_t flags[8];
            ious_[i][j] = diou(bi, bj, nxd_[i][j][0], nxd_[i][j][1], nxd_[i][j][2], flags);

            #pragma unroll
            for (int k = 0; k < 8; k++)
                xflags_[i][j][k] = flags[k];
        }
    });
}

tuple<Tensor, Tensor, Tensor> diou2dr_forward(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    Tensor nxd = torch::empty({boxes1.size(0), boxes2.size(0), 3}, torch::dtype(torch::kByte));
    Tensor xflags = torch::empty({boxes1.size(0), boxes2.size(0), 8}, torch::dtype(torch::kByte));

    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "diou2dr_forward", [&] {
        diou2dr_forward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            ious._cpu_accessor(2),
            nxd._cpu_accessor_t(uint8_t, 3),
            xflags._cpu_accessor_t(uint8_t, 3));
    });
    return make_tuple(ious, nxd, xflags);
}

template <typename scalar_t>
void diou2dr_backward_templated(
    const _CpuAccessor(2) boxes1_,
    const _CpuAccessor(2) boxes2_,
    const _CpuAccessor(2) grad_,
    const _CpuAccessorT(uint8_t, 3) nxd_,
    const _CpuAccessorT(uint8_t, 3) xflags_,
    _CpuAccessor(2) grad_boxes1_,
    _CpuAccessor(2) grad_boxes2_
) {
    using BoxType = Quad2<scalar_t>;
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    parallel_for(0, N*M, 0, [&](int64_t begin, int64_t end)
    {
        for (int64_t nm = begin; nm < end; nm++)
        {
            const int i = nm % N;
            const int j = nm / N;

            BoxType bi = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes1_[i]);
            BoxType bj = _BoxUtilCpu<scalar_t, BoxType>::make_box(boxes2_[j]);

            BoxType grad_i, grad_j;
            grad_i.zero(); grad_j.zero();
            uint8_t flags[8];
            #pragma unroll
            for(int k = 0; k < 8; k++)
                flags[k] = xflags_[i][j][k];

            diou_grad(bi, bj, grad_[i][j], nxd_[i][j][0], nxd_[i][j][1], nxd_[i][j][2], flags, grad_i, grad_j);
            _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes1_[i], grad_i, grad_boxes1_[i]);
            _BoxUtilCpu<scalar_t, BoxType>::make_box_grad(boxes2_[j], grad_j, grad_boxes2_[j]);
        }
    });
}


tuple<Tensor, Tensor> diou2dr_backward(
    const Tensor boxes1, const Tensor boxes2, const Tensor grad,
    const Tensor nxd, const Tensor xflags
) {
    Tensor grad_boxes1 = torch::zeros_like(boxes1);
    Tensor grad_boxes2 = torch::zeros_like(boxes2);
    
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "diou2dr_backward", [&] {
        diou2dr_backward_templated<scalar_t>(
            boxes1._cpu_accessor(2),
            boxes2._cpu_accessor(2),
            grad._cpu_accessor(2),
            nxd._cpu_accessor_t(uint8_t, 3),
            xflags._cpu_accessor_t(uint8_t, 3),
            grad_boxes1._cpu_accessor(2),
            grad_boxes2._cpu_accessor(2));
    });

    return make_tuple(grad_boxes1, grad_boxes2);
}
