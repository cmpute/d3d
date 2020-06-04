#include "d3d/common.h"
#include "d3d/box/nms.h"
#include "d3d/box/geometry.hpp"

using namespace std;
using namespace torch;

// some configurations
typedef int64_t bitvec_t;
constexpr c10::ScalarType bitvec_dtype = torch::kLong;
constexpr int FLAG_BITS = 6; // in the following code, x << FLAG_BITS is the same as x * FLAGS_WIDTH
constexpr int FLAG_WIDTH = 1 << FLAG_BITS;
static_assert(FLAG_WIDTH == sizeof(bitvec_t) * 8, "Inconsistant flag width!");


template <typename scalar_t, typename TBox>
struct _BoxUtil
{ 
    static CUDA_CALLABLE_MEMBER TBox make_box(const _CudaSubAccessor(1) data);
    static CUDA_CALLABLE_MEMBER TBox make_box(const scalar_t* data);
};
template <typename scalar_t>
struct _BoxUtil<scalar_t, Box2>
{
    static CUDA_CALLABLE_MEMBER Box2 make_box(const _CudaSubAccessor(1) data)
    {
        return Box2(data[0], data[1], data[2], data[3], data[4]);
    }
    static CUDA_CALLABLE_MEMBER Box2 make_box(const scalar_t* data)
    {
        return Box2(data[0], data[1], data[2], data[3], data[4]);
    }
};
template <typename scalar_t>
struct _BoxUtil<scalar_t, AABox2>
{
    static CUDA_CALLABLE_MEMBER AABox2 make_box(const _CudaSubAccessor(1) data)
    {
        return Box2(data[0], data[1], data[2], data[3], data[4]).bbox();
    }
    static CUDA_CALLABLE_MEMBER AABox2 make_box(const scalar_t* data)
    {
        return Box2(data[0], data[1], data[2], data[3], data[4]).bbox();
    }
};

template <typename scalar_t, IouType Iou, SupressionType Supression>
__global__ void nms2d_iou_kernel(
    const _CudaAccessor(2) boxes_,
    const _CudaAccessorT(int64_t, 1) order_,
    const scalar_t iou_threshold,
    const scalar_t supression_param, // parameter for supression
    _CudaAccessor(2) iou_coeffs_,
    _CudaAccessorT(bitvec_t, 2) mask_
) {
    using BoxType = typename std::conditional<Iou == IouType::BOX, AABox2, Box2>::type;
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    if (row_start > col_start) return; // calculate only blocks in upper triangle part

    const int row_size = min(boxes_.size(0) - row_start * FLAG_WIDTH, FLAG_WIDTH);
    const int col_size = min(boxes_.size(0) - col_start * FLAG_WIDTH, FLAG_WIDTH);

    __shared__ scalar_t block_boxes[FLAG_WIDTH][5]; // TODO: directly store boxes here. Need to fix Poly2 alignment
                                                    // https://stackoverflow.com/questions/37323053/misaligned-address-in-cuda
                                                    // https://stackoverflow.com/questions/12778949/cuda-memory-alignment
                                                    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
                                                    // or maybe let Poly2 to have dynamic memory with a pointer (use malloc)
                                                    // https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
                                                    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#per-thread-block-allocation
    if (threadIdx.x < col_size)
    {
        #pragma unroll
        for (int i = 0; i < 5; ++i)
        {
            int boxi = order_[FLAG_WIDTH * col_start + threadIdx.x];
            block_boxes[threadIdx.x][i] = boxes_[boxi][i];
        }
    }
    __syncthreads();

    // calculate suppression in this cropped box
    if (threadIdx.x < row_size)
    {
        const int idx = FLAG_WIDTH * row_start + threadIdx.x;
        const int bcur_idx = order_[idx];
        BoxType bcur = _BoxUtil<scalar_t, BoxType>::make_box(boxes_[bcur_idx]);

        int64_t flag = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0; // also calculate only upper part in diagonal blocks
        for (int i = start; i < col_size; i++)
        {
            BoxType bcomp = _BoxUtil<scalar_t, BoxType>::make_box(block_boxes[i]);
            scalar_t iou = bcur.iou(bcomp);
            if (iou > iou_threshold)
                flag |= 1ULL << i; // mark overlap

            switch(Supression)
            {
            case SupressionType::LINEAR:
                iou_coeffs_[idx][i] *= 1 - pow(iou, supression_param); 
                break;
            case SupressionType::GAUSSIAN:
                iou_coeffs_[idx][i] *= exp(-iou * iou / supression_param);
                break;
            }
        }
        mask_[idx][col_start] = flag;
    }
}

template <typename scalar_t, SupressionType Supression>
__global__ void nms_collect_kernel(
    const _CudaAccessorT(int64_t, 1) order_,
    const _CudaAccessorT(bitvec_t, 2) mask_,
    const _CudaAccessor(2) iou_coeffs_,
    _CudaAccessorT(bitvec_t, 1) remv_,
    _CudaAccessor(1) scores_, // original score array
    const float score_threshold,
    _CudaAccessorT(bool, 1) suppressed_ // need to be filled by false
) {
    const int nboxes = mask_.size(0);
    const int nblocks = mask_.size(1);

    for (int i = 0; i < nboxes; i++)
    {
        int block_idx = i / FLAG_WIDTH;
        int thread_idx = i % FLAG_WIDTH;

        if (remv_[block_idx] & (1ULL << thread_idx)) // already suppressed
            suppressed_[order_[i]] = true; // mark
        else // suppress succeeding blocks
            for (int j = block_idx; j < nblocks; j++)
            {
                if (Supression == SupressionType::HARD)
                    remv_[j] |= mask_[i][j]; // process 64 bits simutaneously
                else 
                    for (int k = 0; k < FLAG_WIDTH; k++)
                        if (mask_[i][j] & (1ULL << k))
                        {
                            int i2 = j * FLAG_WIDTH + k; // TODO: use bit shift to speed up
                            if (remv_[j] & 1ULL << k) // already suppressed
                                continue;

                            scores_[i2] *= iou_coeffs_[i][i2];
                            if (scores_[i2] < score_threshold)
                                remv_[j] |= 1ULL << k;
                        }
            }
    }
}

template <typename scalar_t, IouType Iou, SupressionType Supression>
void nms2d_cuda_templated(
    const Tensor boxes, const Tensor order, const Tensor scores,
    const float iou_threshold, const float score_threshold, const float supression_param,
    Tensor suppressed
) {
    const auto device = boxes.device();
    const int nboxes = boxes.sizes().at(0);
    const int nblocks = divup(nboxes, FLAG_WIDTH);

    // This tensor store pairwise IOU flags, rows are continuous while cols are divided by FLAG_WIDTH.
    // It has type int64, but it can act as uint64 in terms of bit operation.
    // Also note that the index in mask is corresponding to the position in `order` tensor.
    Tensor mask = torch::zeros({nboxes, nblocks}, torch::dtype(bitvec_dtype).device(device));
    Tensor iou_coeffs;
    if (Supression == SupressionType::HARD)
        iou_coeffs = torch::zeros({0, 0}, boxes.options());
    else
        iou_coeffs = torch::zeros({nboxes, nboxes}, boxes.options());

    dim3 blocks(nblocks, nblocks);
    dim3 threads(FLAG_WIDTH);

    nms2d_iou_kernel<scalar_t, Iou, Supression><<<blocks, threads>>>(
        boxes._cuda_accessor(2),
        order._cuda_accessor_t(int64_t, 1),
        (scalar_t) iou_threshold,
        (scalar_t) supression_param,
        iou_coeffs._cuda_accessor(2),
        mask._cuda_accessor_t(bitvec_t, 2)
    );

    // temporary tensor for block suppression flags
    auto remv = torch::zeros({nblocks}, torch::dtype(bitvec_dtype).device(device));
    nms_collect_kernel<scalar_t, Supression><<<1, 1>>>(
        order._cuda_accessor_t(int64_t, 1),
        mask._cuda_accessor_t(bitvec_t, 2),
        iou_coeffs._cuda_accessor(2),
        remv._cuda_accessor_t(bitvec_t, 1),
        scores._cuda_accessor(1),
        score_threshold,
        suppressed._cuda_accessor_t(bool, 1)
    );
}

Tensor nms2d_cuda(
    const Tensor boxes, const Tensor scores,
    const IouType iou_type, const SupressionType supression_type,
    const float iou_threshold, const float score_threshold, const float supression_param
) {
    // First filter out boxes with lower scores
    Tensor score_mask = scores > score_threshold;
    Tensor boxes_masked = boxes.index({score_mask});
    Tensor scores_masked = scores.index({score_mask});
    Tensor order = scores_masked.argsort(-1, true);
    Tensor suppressed = torch::zeros({boxes_masked.size(0)}, torch::dtype(torch::kBool).device(boxes.device()));

    AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "nms2d_cuda",
        _NMS_DISPATCH_IOUTYPE(iou_type, _NMS_DISPATCH_SUPRESSTYPE(supression_type, [&] {
            nms2d_cuda_templated<scalar_t, Iou, Supression>(
                boxes_masked, order, scores_masked,
                iou_threshold, score_threshold, supression_param,
                suppressed);
        }))
    );

    // TODO: return ~score_mask & suppressed
    return suppressed;
}
