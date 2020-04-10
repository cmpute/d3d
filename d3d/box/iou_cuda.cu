#include "d3d/common.h"
#include "d3d/box/iou.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

template <typename scalar_t>
__global__ void rbox_2d_iou_kernel(
    const _PackedAccessor(2) boxes1,
    const _PackedAccessor(2) boxes2,
    _PackedAccessor(2) ious
) {
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = nm / boxes1.size(0);
    const int j = nm % boxes1.size(0);
    
    Box2 bi(boxes1[i][0], boxes1[i][1], boxes1[i][2],
        boxes1[i][3], boxes1[i][4]);
    Box2 bj(boxes2[j][0], boxes2[j][1], boxes2[j][2],
        boxes2[j][3], boxes2[j][4]);
    ious[i][j] = bi.iou(bj);
}

Tensor rbox_2d_iou_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)},
        torch::dtype(torch::kFloat32).device(boxes1.device()));
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = DivUp(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.type(), "rbox_2d_iou_cuda", ([&] {
        rbox_2d_iou_kernel<scalar_t><<<blocks, threads>>>(
        boxes1._packed_accessor(2),
        boxes2._packed_accessor(2),
        ious._packed_accessor(2));
    }));

    return ious;
}

template <typename scalar_t>
__global__ void box_2d_iou_kernel(
    const _PackedAccessor(2) boxes1,
    const _PackedAccessor(2) boxes2,
    _PackedAccessor(2) ious
) {
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = nm / boxes1.size(0);
    const int j = nm % boxes1.size(0);
    
    AABox2 bi = Box2(boxes1[i][0], boxes1[i][1], boxes1[i][2],
        boxes1[i][3], boxes1[i][4]).bbox();
    AABox2 bj = Box2(boxes2[j][0], boxes2[j][1], boxes2[j][2],
        boxes2[j][3], boxes2[j][4]).bbox();
    ious[i][j] = bi.iou(bj);
}

Tensor box_2d_iou_cuda(
    const Tensor boxes1, const Tensor boxes2
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)},
        torch::dtype(torch::kFloat32).device(boxes1.device()));
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = DivUp(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.type(), "rbox_2d_iou_cuda", ([&] {
        box_2d_iou_kernel<scalar_t><<<blocks, threads>>>(
        boxes1._packed_accessor(2),
        boxes2._packed_accessor(2),
        ious._packed_accessor(2));
    }));

    return ious;
}
