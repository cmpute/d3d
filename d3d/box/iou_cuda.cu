#include "d3d/common.h"
#include "d3d/box/iou.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

template <typename scalar_t>
__global__ void rbox_2d_iou_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_
) {
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = nm % boxes1_.size(0);
    const int j = nm / boxes1_.size(0);
    
    // FIXME: make one of the box shared among thread? need to benchmark this
    Box2 bi(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
        boxes1_[i][3], boxes1_[i][4]);
    Box2 bj(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
        boxes2_[j][3], boxes2_[j][4]);
    ious_[i][j] = bi.iou(bj);
}

Tensor rbox_2d_iou_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)},
        torch::dtype(torch::kFloat32).device(boxes1.device()));
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "rbox_2d_iou_cuda", ([&] {
        rbox_2d_iou_kernel<scalar_t><<<blocks, threads>>>(
        boxes1._cuda_accessor(2),
        boxes2._cuda_accessor(2),
        ious._cuda_accessor(2));
    }));

    return ious;
}

template <typename scalar_t>
__global__ void box_2d_iou_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_
) {
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = nm % boxes1_.size(0);
    const int j = nm / boxes1_.size(0);
    
    AABox2 bi = Box2(boxes1_[i][0], boxes1_[i][1], boxes1_[i][2],
        boxes1_[i][3], boxes1_[i][4]).bbox();
    AABox2 bj = Box2(boxes2_[j][0], boxes2_[j][1], boxes2_[j][2],
        boxes2_[j][3], boxes2_[j][4]).bbox();
    ious_[i][j] = bi.iou(bj);
}

Tensor box_2d_iou_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)}, boxes1.options());
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "box_2d_iou_cuda", ([&] {
        box_2d_iou_kernel<scalar_t><<<blocks, threads>>>(
        boxes1._cuda_accessor(2),
        boxes2._cuda_accessor(2),
        ious._cuda_accessor(2));
    }));

    return ious;
}
