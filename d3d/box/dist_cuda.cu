#include "d3d/common.h"
#include "d3d/box/iou.h"
#include "d3d/box/utils.cuh"

using namespace std;
using namespace torch;
using namespace dgal;

template <typename scalar_t>
__global__ void pdist2dr_forward_kernel(
    const _CudaAccessor(2) points_,
    const _CudaAccessor(2) boxes_,
    _CudaAccessor(2) distance_,
    _CudaAccessorT(uint8_t, 2) iedge_
) {
    using BoxType = Quad2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes_.size(0);
    const auto M = points_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType b = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes_[i]);
        Point2<scalar_t> p {.x=points_[j][0], .y=points_[j][1]};
        distance_[i][j] = distance(b, p, iedge_[i][j]);
    }
}

tuple<Tensor, Tensor> pdist2dr_forward_cuda(
    const Tensor points, const Tensor boxes
) {
    Tensor distance = torch::empty({boxes.size(0), points.size(0)}, points.options());
    Tensor iedge = torch::empty({boxes.size(0), points.size(0)}, torch::dtype(torch::kByte).device(points.device()));

    const int total_ops = boxes.size(0) * points.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);

    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "pdist2dr_forward", [&] {
        pdist2dr_forward_kernel<scalar_t><<<blocks, threads>>>(
            points._cuda_accessor(2),
            boxes._cuda_accessor(2),
            distance._cuda_accessor(2),
            iedge._cuda_accessor_t(uint8_t, 2));
    });
    return make_tuple(distance, iedge);
}

template <typename scalar_t>
__global__ void pdist2dr_backward_kernel(
    const _CudaAccessor(2) points_,
    const _CudaAccessor(2) boxes_,
    const _CudaAccessor(2) grad_,
    _CudaAccessor(2) grad_boxes_,
    _CudaAccessor(2) grad_points_,
    _CudaAccessorT(uint8_t, 2) iedge_
) {
    using BoxType = Quad2<scalar_t>;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes_.size(0);
    const auto M = points_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;

        BoxType b = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes_[i]);
        BoxType grad_b; grad_b.zero();
        Point2<scalar_t> p {.x=points_[j][0], .y=points_[j][1]};
        Point2<scalar_t> grad_p;
        distance_grad(b, p, grad_[i][j], grad_b, grad_p, iedge_[i][j]);

        _BoxUtilCuda<scalar_t, BoxType>::make_box_grad(boxes_[i], grad_b, grad_boxes_[i]);
        grad_points_[j][0] += grad_p.x;
        grad_points_[j][1] += grad_p.y;
    }
}

tuple<Tensor, Tensor> pdist2dr_backward_cuda(
    const Tensor points, const Tensor boxes, const Tensor grad,
    const Tensor iedge
) {
    Tensor grad_boxes = torch::zeros_like(boxes);
    Tensor grad_points = torch::zeros_like(points);

    const int total_ops = boxes.size(0) * points.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "pdist2dr_backward", [&] {
        pdist2dr_backward_kernel<scalar_t><<<blocks, threads>>>(
            points._cuda_accessor(2),
            boxes._cuda_accessor(2),
            grad._cuda_accessor(2),
            grad_boxes._cuda_accessor(2),
            grad_points._cuda_accessor(2),
            iedge._cuda_accessor_t(uint8_t, 2));
    });

    return make_tuple(grad_boxes, grad_points);
}
