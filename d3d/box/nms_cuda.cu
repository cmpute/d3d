#include "d3d/common.h"
#include "d3d/box/nms.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

constexpr int FLAG_WIDTH = sizeof(int64_t) * 8;

template <typename scalar_t>
__global__ void rbox_2d_nms_kernel(
    const _PackedAccessor(2) boxes,
    const _PackedAccessorT(int64_t, 1) order,
    const scalar_t threshold,
    _PackedAccessorT(int64_t, 2) mask
) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    if (row_start > col_start) return; // calculate only blocks in upper triangle part

    const int row_size = min(boxes.size(0) - row_start * FLAG_WIDTH, FLAG_WIDTH);
    const int col_size = min(boxes.size(0) - col_start * FLAG_WIDTH, FLAG_WIDTH);

    __shared__ scalar_t block_boxes[FLAG_WIDTH][5]; // XXX: find a way to declare Box2 object here directly
    if (threadIdx.x < col_size)
    {
        #pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            int boxi = order[FLAG_WIDTH * col_start + threadIdx.x];
            block_boxes[threadIdx.x][i] = boxes[boxi][i];
        }
    }
    __syncthreads();

    // calculate suppression in this cropped box
    if (threadIdx.x < row_size)
    {
        const int idx = FLAG_WIDTH * row_start + threadIdx.x;
        const int bcur_idx = order[idx];
        Box2 bcur(boxes[bcur_idx][0], boxes[bcur_idx][1], boxes[bcur_idx][2],
            boxes[bcur_idx][3], boxes[bcur_idx][4]);

        int64_t flag = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0; // also calculate only upper part in diagonal blocks
        for (int i = start; i < col_size; i++)
        {
            Box2 bcomp(block_boxes[i][0], block_boxes[i][1], block_boxes[i][2],
                block_boxes[i][3], block_boxes[i][4]);
            if (bcur.iou(bcomp) > threshold)
                flag |= 1ULL << i;
        }
        mask[idx][col_start] = flag;
    }
}

template <typename scalar_t>
__global__ void box_2d_nms_kernel(
    const _PackedAccessor(2) boxes,
    const _PackedAccessorT(int64_t, 1) order,
    const scalar_t threshold,
    _PackedAccessorT(int64_t, 2) mask
) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    if (row_start > col_start) return; // calculate only blocks in upper triangle part

    const int row_size = min(boxes.size(0) - row_start * FLAG_WIDTH, FLAG_WIDTH);
    const int col_size = min(boxes.size(0) - col_start * FLAG_WIDTH, FLAG_WIDTH);

    __shared__ scalar_t block_boxes[FLAG_WIDTH][5]; // XXX: find a way to declare Box2 object here directly
    if (threadIdx.x < col_size)
    {
        #pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            int boxi = order[FLAG_WIDTH * col_start + threadIdx.x];
            block_boxes[threadIdx.x][i] = boxes[boxi][i];
        }
    }
    __syncthreads();

    // calculate suppression in this cropped box
    if (threadIdx.x < row_size)
    {
        const int idx = FLAG_WIDTH * row_start + threadIdx.x;
        const int bcur_idx = order[idx];
        AABox2 bcur = Box2(boxes[bcur_idx][0], boxes[bcur_idx][1], boxes[bcur_idx][2],
            boxes[bcur_idx][3], boxes[bcur_idx][4]).bbox();

        int64_t flag = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0; // also calculate only upper part in diagonal blocks
        for (int i = start; i < col_size; i++)
        {
            AABox2 bcomp = Box2(block_boxes[i][0], block_boxes[i][1], block_boxes[i][2],
                block_boxes[i][3], block_boxes[i][4]).bbox();
            if (bcur.iou(bcomp) > threshold)
                flag |= 1ULL << i;
        }
        mask[idx][col_start] = flag;
    }
}

__global__ void nms_collect(
    const _PackedAccessorT(int64_t, 1) order,
    const _PackedAccessorT(int64_t, 2) mask,
    _PackedAccessorT(int64_t, 1) remv,
    _PackedAccessorT(bool, 1) suppressed // need to be filled by false
) {
    const int nboxes = mask.size(0);
    const int nblocks = mask.size(1);

    for (int i = 0; i < nboxes; i++)
    {
        int block_idx = i / FLAG_WIDTH;
        int thread_idx = i % FLAG_WIDTH;

        if (remv[block_idx] & (1ULL << thread_idx))
            suppressed[order[i]] = true;
        else // suppress succeeding blocks
            for (int j = block_idx; j < nblocks; j++)
                remv[j] |= mask[i][j];
    }
}

template <int KernelType> // 0: box_iou, 1: rbox_iou
Tensor nms2d_cuda_template(
  const Tensor boxes, const Tensor order, const float threshold
) {
    auto device = boxes.device();
    const int nboxes = boxes.sizes().at(0);
    const int nblocks = DivUp(nboxes, FLAG_WIDTH);

    // This tensor store pairwise IOU result, rows are continuous while cols are divided by FLAG_WIDTH.
    // It has type int64, but it can act as uint64 in terms of bit operation.
    // Also note that the index in mask is corresponding to the position in `order` tensor.
    auto mask = torch::zeros({nboxes, nblocks}, torch::dtype(torch::kLong).device(device));

    dim3 blocks(nblocks, nblocks);
    dim3 threads(FLAG_WIDTH);

    switch(KernelType)
    {
        default:
        case 0:
            AT_DISPATCH_FLOATING_TYPES(boxes.type(), "box_2d_nms_kernel", ([&] {
                box_2d_nms_kernel<<<blocks, threads>>>(
                    boxes._packed_accessor(2),
                    order._packed_accessor_typed(int64_t, 1),
                    (scalar_t) threshold,
                    mask._packed_accessor_typed(int64_t, 2));
            }));
            break;
        case 1:
            AT_DISPATCH_FLOATING_TYPES(boxes.type(), "rbox_2d_nms_kernel", ([&] {
                rbox_2d_nms_kernel<<<blocks, threads>>>(
                    boxes._packed_accessor(2),
                    order._packed_accessor_typed(int64_t, 1),
                    (scalar_t) threshold,
                    mask._packed_accessor_typed(int64_t, 2));
            }));
            break;
    }

    auto remv = torch::zeros({nblocks}, torch::dtype(torch::kLong).device(device)); // suppression flags
    auto suppressed = torch::zeros({nboxes}, torch::dtype(torch::kBool).device(device));
    nms_collect<<<1, 1>>>(
        order._packed_accessor_typed(int64_t, 1),
        mask._packed_accessor_typed(int64_t, 2),
        remv._packed_accessor_typed(int64_t, 1),
        suppressed._packed_accessor_typed(bool, 1));

    return suppressed;
}

Tensor box_2d_nms_cuda(
    const Tensor boxes, const Tensor order, const float threshold
) {
    return nms2d_cuda_template<0>(boxes, order, threshold);
}

Tensor rbox_2d_nms_cuda(
    const Tensor boxes, const Tensor order, const float threshold
) {
    return nms2d_cuda_template<1>(boxes, order, threshold);
}
