#include "d3d/common.h"
#include "d3d/box/nms.h"
#include "d3d/box/utils.cuh"

using namespace std;
using namespace torch;

// some configurations
typedef int64_t bitvec_t;
constexpr c10::ScalarType bitvec_dtype = torch::kLong;
constexpr int FLAG_BITS = 6; // in the following code, x << FLAG_BITS is the same as x * FLAGS_WIDTH
constexpr int FLAG_WIDTH = 1 << FLAG_BITS;
static_assert(FLAG_WIDTH == sizeof(bitvec_t) * 8, "Inconsistant flag width!");

template <typename scalar_t, IouType Iou, SupressionType Supression>
__global__ void nms2d_iou_kernel(
    const _CudaAccessor(2) boxes_,
    const _CudaAccessorT(int64_t, 1) order_,
    const scalar_t iou_threshold,
    const scalar_t supression_param, // parameter for supression
    _CudaAccessor(2) iou_coeffs_, // store suppression coefficients
    _CudaAccessorT(bitvec_t, 2) iou_mask_ // store suppression masks
) {
    using BoxType = typename std::conditional<Iou == IouType::BOX, AABox2f, Box2f>::type;
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    if (row_start > col_start) return; // calculate only blocks in upper triangle part

    const int row_size = min(boxes_.size(0) - (row_start << FLAG_BITS), FLAG_WIDTH);
    const int col_size = min(boxes_.size(0) - (col_start << FLAG_BITS), FLAG_WIDTH);
    __shared__ scalar_t block_boxes[FLAG_WIDTH][5];

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
        const int idx = (row_start << FLAG_BITS) + threadIdx.x;
        const int bcur_idx = order_[idx];
        BoxType bcur = _BoxUtilCuda<scalar_t, BoxType>::make_box(boxes_[bcur_idx]);

        int64_t flag = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0; // also calculate only upper part in diagonal blocks
        for (int i = start; i < col_size; i++)
        {
            BoxType bcomp = _BoxUtilCuda<scalar_t, BoxType>::make_box(block_boxes[i]);
            scalar_t iou = bcur.iou(bcomp);
            if (iou <= iou_threshold)
                continue;

            switch(Supression)
            {
            case SupressionType::HARD:
                flag |= 1ULL << i; // mark overlap
                break;
            case SupressionType::LINEAR:
                iou_coeffs_[bcur_idx][order_[i]] = 1 - pow(iou, supression_param); 
                break;
            case SupressionType::GAUSSIAN:
                iou_coeffs_[bcur_idx][order_[i]] = exp(-iou * iou / supression_param);
                break;
            }
        }
        if (Supression == SupressionType::HARD)
            iou_mask_[idx][col_start] = flag;
    }
}

template <typename scalar_t>
__global__ void nms_collect_kernel(
    const _CudaAccessorT(bitvec_t, 2) iou_mask_,
    const _CudaAccessorT(int64_t, 1) order_,
    _CudaAccessorT(bool, 1) suppressed_ // need to be filled by false
) {
    const int nboxes = iou_mask_.size(0);
    const int nblocks = iou_mask_.size(1);

    // temporary tensor for block suppression flags
    bitvec_t *remv = new bitvec_t[nblocks];
    for (int i = 0; i < nblocks; i++) remv[i] = 0;

    // main loop
    for (int i = 0; i < nboxes; i++)
    {
        int block_idx = i >> FLAG_BITS;
        int thread_idx = i & (FLAG_WIDTH-1);

        if (remv[block_idx] & (1ULL << thread_idx)) // already suppressed
            suppressed_[order_[i]] = true; // mark
        else // suppress succeeding blocks
            for (int j = block_idx; j < nblocks; j++)
                remv[j] |= iou_mask_[i][j]; // process 64 bits simutaneously
    }

    delete[] remv;
}

template <typename scalar_t>
__global__ void soft_nms_collect_kernel(
    const _CudaAccessor(2) iou_coeffs_,
    _CudaAccessorT(int64_t, 1) order_,
    _CudaAccessor(1) scores_, // original score array
    const float score_threshold,
    _CudaAccessorT(bool, 1) suppressed_ // need to be filled by false
) {
    const int N = scores_.size(0);
    for (int _i = 0; _i < N; _i++)
    {
        int i = order_[_i];
        if (suppressed_[i]) // for soft-nms, remaining part are all suppressed
            break;

        // suppress following boxes with lower score
        for (int _j = _i + 1; _j < N; _j++)
        {
            int j = order_[_j];
            if (iou_coeffs_[i][j] > 0)
            {
                scores_[j] *= iou_coeffs_[i][j];
                suppressed_[j] = scores_[j] < score_threshold;
            }
        }

        // following part is the same as CPU version
        // find suppression block start
        int S = N - 1;
        while (S > _i && !suppressed_[order_[S]]) S--;

        // Sort the scores again with simple insertion sort and put suppressed indices into back
        for (int _j = S - 1; _j > _i; _j--)
        {
            int j = order_[_j];
            int _k = _j + 1;
            while (_k < S && (suppressed_[j] || // suppression is like score = 0
                scores_[order_[_k]] > scores_[j]))
            {
                order_[_k-1] = order_[_k];
                _k++;
            }
            order_[_k - 1] = j;
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

    // iou_mask stores pairwise IOU flags, rows are continuous while cols are divided by FLAG_WIDTH.
    // It has type int64, but it can act as uint64 in terms of bit operation.
    // Also note that the index in iou_mask is corresponding to the sorted position in `order` tensor.
    //
    // iou_coeffs stores suppression coefficients in SoftNMS. If coefficient is zero, then the boxes
    // don't overlap. The index in iou_mask is corresponding to the original position in `scores` tensor.
    Tensor iou_mask, iou_coeffs;
    auto bitvec_options = torch::dtype(bitvec_dtype).device(device);
    if (Supression == SupressionType::HARD)
    {
        iou_coeffs = torch::empty({0, 0}, boxes.options());
        iou_mask = torch::zeros({nboxes, nblocks}, bitvec_options);
    }
    else
    {
        iou_coeffs = torch::zeros({nboxes, nboxes}, boxes.options());
        iou_mask = torch::empty({0, 0}, bitvec_options);
    }

    dim3 blocks(nblocks, nblocks);
    dim3 threads(FLAG_WIDTH);

    // calculate iou
    nms2d_iou_kernel<scalar_t, Iou, Supression><<<blocks, threads>>>(
        boxes._cuda_accessor(2),
        order._cuda_accessor_t(int64_t, 1),
        (scalar_t) iou_threshold,
        (scalar_t) supression_param,
        iou_coeffs._cuda_accessor(2),
        iou_mask._cuda_accessor_t(bitvec_t, 2)
    );

    // do suppression
    if (Supression == SupressionType::HARD)
        nms_collect_kernel<scalar_t><<<1, 1>>>(
            iou_mask._cuda_accessor_t(bitvec_t, 2),
            order._cuda_accessor_t(int64_t, 1),
            suppressed._cuda_accessor_t(bool, 1)
        );
    else
        soft_nms_collect_kernel<scalar_t><<<1, 1>>>(
            iou_coeffs._cuda_accessor(2),
            order._cuda_accessor_t(int64_t, 1),
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
    Tensor order_masked = scores_masked.argsort(-1, true);
    Tensor suppressed_masked = torch::zeros({boxes_masked.size(0)}, torch::dtype(torch::kBool).device(boxes.device()));

    // launch NMS kernels
    AT_DISPATCH_FLOATING_TYPES(boxes.scalar_type(), "nms2d_cuda",
        _NMS_DISPATCH_IOUTYPE(iou_type, _NMS_DISPATCH_SUPRESSTYPE(supression_type, [&] {
            nms2d_cuda_templated<scalar_t, Iou, Supression>(
                boxes_masked, order_masked, scores_masked,
                iou_threshold, score_threshold, supression_param,
                suppressed_masked);
        }))
    );

    // combine suppression mask with score mask
    Tensor suppressed_idx = torch::where(score_mask)[0].index({suppressed_masked});
    Tensor suppressed = score_mask.logical_not();
    suppressed.index_fill_(0, suppressed_idx, true);
    return suppressed;
}
