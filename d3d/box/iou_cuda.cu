#include "d3d/common.h"
#include "d3d/box/iou.h"
#include "d3d/box/utils.cuh"

using namespace std;
using namespace torch;

template <typename scalar_t, IouType Iou>
__global__ void iou2d_kernel(
    const _CudaAccessor(2) boxes1_,
    const _CudaAccessor(2) boxes2_,
    _CudaAccessor(2) ious_
) {
    using BoxType = typename std::conditional<Iou == IouType::BOX, AABox2f, Poly2f>::type;
    const int nm = blockIdx.x * blockDim.x + threadIdx.x;    
    const auto N = boxes1_.size(0);
    const auto M = boxes2_.size(0);

    if (nm < N*M)
    {
        const int i = nm % N;
        const int j = nm / N;
        
        // FIXME: make one of the box shared among threads? need to benchmark this
        BoxType bi = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes1_[i]);
        BoxType bj = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes2_[j]);
        ious_[i][j] = bi.iou(bj);
    }
}

Tensor iou2d_cuda(
    const Tensor boxes1, const Tensor boxes2, const IouType iou_type
) {
    Tensor ious = torch::empty({boxes1.size(0), boxes2.size(0)},
        torch::dtype(torch::kFloat32).device(boxes1.device()));
    const int total_ops = boxes1.size(0) * boxes2.size(0);
    const int threads = THREADS_COUNT;
    const int blocks = divup(total_ops, threads);
  
    AT_DISPATCH_FLOATING_TYPES(boxes1.scalar_type(), "iou2d_cuda", _NMS_DISPATCH_IOUTYPE(iou_type, [&] {
        iou2d_kernel<scalar_t, Iou><<<blocks, threads>>>(
        boxes1._cuda_accessor(2),
        boxes2._cuda_accessor(2),
        ious._cuda_accessor(2));
    }));

    return ious;
}
