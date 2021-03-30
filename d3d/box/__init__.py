import numpy as np
import torch

from .box_impl import (cuda_available,
    iou2d_forward,
    iou2d_backward,
    iou2dr_forward,
    iou2dr_backward,
    giou2dr_forward,
    giou2dr_backward,
    diou2dr_forward,
    diou2dr_backward,
    pdist2dr_forward,
    pdist2dr_backward,
    nms2d as nms2d_cc,
    crop_2dr as crop_2dr_cc,
    IouType, SupressionType
)

if cuda_available:
    from .box_impl import (
        iou2d_forward_cuda,
        iou2d_backward_cuda,
        iou2dr_forward_cuda,
        iou2dr_backward_cuda,
        giou2dr_forward_cuda,
        giou2dr_backward_cuda,
        diou2dr_forward_cuda,
        diou2dr_backward_cuda,
        pdist2dr_forward_cuda,
        pdist2dr_backward_cuda,
        nms2d_cuda,
    )

# TODO: separate iou and iou loss (the latter one is the diagonal result of previous one)
#       and it seems that we need the backward part only for the iou loss

class Iou2D(torch.autograd.Function):
    '''
    Differentiable axis aligned IoU function for 2D boxes
    '''
    @staticmethod
    def forward(ctx, boxes1, boxes2):
        ctx.save_for_backward(boxes1, boxes2)
        if boxes1.is_cuda and boxes2.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return iou2d_forward_cuda(boxes1, boxes2)
        else:
            return iou2d_forward(boxes1, boxes2)

    @staticmethod
    def backward(ctx, grad):
        boxes1, boxes2 = ctx.saved_tensors
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return iou2d_backward_cuda(boxes1, boxes2, grad)
        else:
            return iou2d_backward(boxes1, boxes2, grad)

class Iou2DR(torch.autograd.Function):
    '''
    Differentiable rotated IoU function for 2D boxes
    '''
    @staticmethod
    def forward(ctx, boxes1, boxes2):
        if boxes1.is_cuda and boxes2.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            ious, nx, xflags = iou2dr_forward_cuda(boxes1, boxes2)
        else:
            ious, nx, xflags = iou2dr_forward(boxes1, boxes2)
        ctx.save_for_backward(boxes1, boxes2, nx, xflags)
        return ious

    @staticmethod
    def backward(ctx, grad):
        boxes1, boxes2, nx, xflags = ctx.saved_tensors
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return iou2dr_backward_cuda(boxes1, boxes2, grad, nx, xflags)
        else:
            return iou2dr_backward(boxes1, boxes2, grad, nx, xflags)

class GIou2DR(torch.autograd.Function):
    '''
    Differentiable rotated GIoU function for 2D boxes
    '''
    @staticmethod
    def forward(ctx, boxes1, boxes2):
        if boxes1.is_cuda and boxes2.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            ious, nxm, flags = giou2dr_forward_cuda(boxes1, boxes2)
        else:
            ious, nxm, flags = giou2dr_forward(boxes1, boxes2)
        ctx.save_for_backward(boxes1, boxes2, nxm, flags)
        return ious
        
    @staticmethod
    def backward(ctx, grad):
        boxes1, boxes2, nxm, flags = ctx.saved_tensors
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return giou2dr_backward_cuda(boxes1, boxes2, grad, nxm, flags)
        else:
            return giou2dr_backward(boxes1, boxes2, grad, nxm, flags)

class DIou2DR(torch.autograd.Function):
    '''
    Differentiable rotated DIoU function for 2D boxes
    '''
    @staticmethod
    def forward(ctx, boxes1, boxes2):
        if boxes1.is_cuda and boxes2.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            ious, nxd, flags = diou2dr_forward_cuda(boxes1, boxes2)
        else:
            ious, nxd, flags = diou2dr_forward(boxes1, boxes2)
        ctx.save_for_backward(boxes1, boxes2, nxd, flags)
        return ious
        
    @staticmethod
    def backward(ctx, grad):
        boxes1, boxes2, nxd, flags = ctx.saved_tensors
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return diou2dr_backward_cuda(boxes1, boxes2, grad, nxd, flags)
        else:
            return diou2dr_backward(boxes1, boxes2, grad, nxd, flags)

class PDist2DR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, boxes):
        if boxes.is_cuda and points.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            dist, iedge = pdist2dr_forward_cuda(boxes, points)
        else:
            dist, iedge = pdist2dr_forward(boxes, points)
        ctx.save_for_backward(points, boxes, iedge)
        return dist

    @staticmethod
    def backward(ctx, grad):
        points, boxes, iedge = ctx.saved_tensors
        if grad.is_cuda:
            assert cuda_available, "d3d was not built with CUDA support!"
            return pdist2dr_backward_cuda(boxes, points, grad, iedge)
        else:
            return pdist2dr_backward(boxes, points, grad, iedge)

def seg1d_iou(seg1, seg2):
    '''
    Calculate IoU of 1D segments
    The input should be n*2, where the last dim is [center, width]

    :param boxes1: Input segments, shape is Nx2
    :param boxes2: Input segments, shape is Nx2
    '''
    assert torch.all(seg1[:,1] > 0)
    assert torch.all(seg2[:,1] > 0)

    dseg1 = seg1[:,1] / 2
    dseg2 = seg1[:,1] / 2

    s1max = seg1[:,0] + dseg1
    s1min = seg1[:,0] - dseg1
    s2max = seg2[:,0] + dseg2
    s2min = seg2[:,0] - dseg2
    
    imax = torch.where(s1max > s2max, s2max, s1max)
    imin = torch.where(s1min < s2min, s2min, s1min)
    umax = torch.where(s1max > s2max, s1max, s2max)
    umin = torch.where(s1min < s2min, s1min, s2min)
    
    i = torch.clamp_min(imax - imin, 0)
    u = torch.clamp_min(umax - umin, 1e-6)
    return i / u

def box2d_iou(boxes1, boxes2, method="box", precise=True):
    '''
    Differentiable IoU on axis-aligned or rotated 2D boxes

    :param boxes1: Input boxes, shape is N x 5 (x,y,w,h,r)
    :param boxes2: Input boxes, shape is N x 5 (x,y,w,h,r)
    :param method: 'box' - axis-aligned box, 'rbox' - rotated box,
        'grbox' - giou for rotated box, 'drbox' - diou for rotated box
    :param precise: force using double precision to calculate iou
    '''
    convert_numpy = False
    if isinstance(boxes1, np.ndarray):
        assert isinstance(boxes2, np.ndarray), "Input should be both numpy tensor or pytorch tensor!"
        boxes1 = torch.from_numpy(boxes1)
        boxes2 = torch.from_numpy(boxes2)
        convert_numpy = True

    otype = boxes1.dtype
    if precise:
        boxes1 = boxes1.to(torch.float64)
        boxes2 = boxes2.to(torch.float64)

    if len(boxes1.shape) != 2 or len(boxes2.shape) != 2:
        raise ValueError("Input of rbox_2d_iou should be Nx2 tensors!")
    if boxes1.shape[1] != 5 or boxes2.shape[1] != 5:
        raise ValueError("Input boxes should have 5 fields: x, y, w, h, r")

    iou_type = getattr(IouType, method.upper())
    if iou_type == IouType.BOX:
        impl = Iou2D
    elif iou_type == IouType.RBOX:
        impl = Iou2DR
    elif iou_type == IouType.GRBOX:
        impl = GIou2DR
    elif iou_type == IouType.DRBOX:
        impl = DIou2DR
    else:
        raise ValueError("Unrecognized iou type!")
    result = impl.apply(boxes1, boxes2)

    if precise:
        result = result.to(otype)
    if convert_numpy:
        return result.numpy()
    return result

def box2d_nms(boxes, scores, iou_method="box", supression_method="hard",
    iou_threshold=0, score_threshold=0, supression_param=0, precise=True):
    '''
    NMS on axis-aligned or rotated 2D boxes

    :param method: 'box' - axis-aligned box, 'rbox' - rotated box
    :param precise: force using double precision to calculate iou
    :param iou_threshold: IoU threshold for two boxes to be considered as overlapped
    :param score_threshold: Minimum score for a box to be considered as valid
    :param suppression_param: Type of suppression. {0: hard, 1: linear, 2: gaussian}. See reference below for details ..

        Soft-NMS: Bodla, Navaneeth, et al. “Soft-NMS–improving object detection with one line of code.” Proceedings of the IEEE international conference on computer vision. 2017.

    '''
    convert_numpy = False
    if isinstance(boxes, np.ndarray):
        assert isinstance(scores, np.ndarray), "Input should be both numpy tensor or pytorch tensor!"
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        convert_numpy = True

    if precise:
        boxes = boxes.to(torch.float64)
        scores = scores.to(torch.float64)

    if len(boxes) != len(scores):
        raise ValueError("Numbers of boxes and scores are inconsistent!")
    if len(scores.shape) == 2:
        scores = scores.max(axis=1).values

    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.bool)

    iou_type = getattr(IouType, iou_method.upper())
    supression_type = getattr(SupressionType, supression_method.upper())

    if boxes.is_cuda and scores.is_cuda:
        assert cuda_available, "d3d was not built with CUDA support!"
        impl = nms2d_cuda
    else:
        impl = nms2d_cc

    suppressed = impl(boxes, scores,
        iou_type, supression_type,
        iou_threshold, score_threshold, supression_param
    )
    mask = ~suppressed

    if convert_numpy:
        return mask.numpy()
    return mask

def box2dr_crop(points, boxes):
    '''
    Crop point points points out given rotated boxes.
    The result is a list of indices tensor where each tensor is corresponding to indices of points lying in the box

    :param points: The input point points, shape: N x 2
    :param boxes: Input boxes array, shape: M x 5
    '''
    result = crop_2dr_cc(points, boxes)
    return result

def box3dp_crop(points, boxes, project_axis=2):
    '''
    Crop point points points out given rotated boxes with boxes projected to given axis

    :param points: The input point points, shape: N x 3
    :param boxes: Input boxes array, shape: M x 7
    :param project_axis: Axis for the box to be projected to. {0: x, 1: y, 2: z}
    '''

    if project_axis == 0:
        points_2d = points[:, [1,2]]
        boxes_2d = boxes[:, [1,2,4,5,6]]
    elif project_axis == 1:
        points_2d = points[:, [0,2]]
        boxes_2d = boxes[:, [0,2,3,5,6]]
    elif project_axis == 2:
        points_2d = points[:, [0,1]]
        boxes_2d = boxes[:, [0,1,3,4,6]]
    else:
        raise ValueError("The projection axis can only be 0-x, 1-y and 2-z!")
    mask_2d = crop_2dr_cc(points_2d, boxes_2d)

    points_p = points[:, [project_axis]].t()
    boxes_p = boxes[:, [project_axis]]
    boxes_pd = boxes[:, [3+project_axis]] / 2
    mask_p = (points_p - boxes_pd < boxes_p) & (boxes_p < points_p + boxes_pd)
    return mask_2d & mask_p

def seg1d_pdist(points, segs):
    '''
    Calculate distance from points to segments
    The input segments should be n*2, where the last dim is [center, width]

    :param points: Input points, shape is Nx1
    :param segs: Input segments, shape is Nx2
    '''
    assert torch.all(segs[:,1] > 0)
    dsegs = segs[:,1] / 2

    smax = segs[:,0] + dsegs
    smin = segs[:,0] - dsegs

    return torch.where(points > segs[:,0], smax - points, points - smin)

def box2dr_pdist(points, boxes, method="rbox"):
    '''
    Calculate signed distance from points to 2d boxes (surfaces)

    :param points: target points, shape: N x 2
    :param boxes: target boxes, shape: M x 5
    :param method: 'rbox' - rotated box
    '''
    if len(boxes.shape) != 2:
        raise ValueError("Input boxes should be Nx2 tensors!")
    if boxes.shape[1] != 5:
        raise ValueError("Input boxes should have 5 fields: x, y, w, h, r")
    if method != "rbox":
        raise ValueError("Only supported rotated boxes by now!")

    result = PDist2DR.apply(points, boxes)
    return result

def box3dr_pdist(points, boxes, project_axis=2):
    '''
    Calculate signed distance from points to 3d boxes (surfaces)

    :param points: target points, shape: N x 3
    :param boxes: target boxes, shape: M x 7
    :param project_axis: Axis for the box to be projected to. {0: x, 1: y, 2: z}
    '''
    if project_axis == 0:
        points_2d = points[:, [1,2]]
        boxes_2d = boxes[:, [1,2,4,5,6]]
    elif project_axis == 1:
        points_2d = points[:, [0,2]]
        boxes_2d = boxes[:, [0,2,3,5,6]]
    elif project_axis == 2:
        points_2d = points[:, [0,1]]
        boxes_2d = boxes[:, [0,1,3,4,6]]
    else:
        raise ValueError("The projection axis can only be 0-x, 1-y and 2-z!")
    dist_2d = box2dr_pdist(points_2d, boxes_2d)

    points_p = points[:, [project_axis]]
    segs_p = boxes[:, [project_axis, 3+project_axis]]
    dist_p = seg1d_pdist(points_p, segs_p)

    dist = torch.where(
        dist_p > 0,
        torch.where(dist_2d > 0, torch.min(dist_p, dist_2d), dist_2d),
        torch.where(dist_2d > 0, dist_p, -torch.sqrt(dist_2d.square() + dist_p.square()))
    )
    return dist
