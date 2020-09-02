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
    nms2d as nms2d_cc,
    rbox_2d_crop as rbox_2d_crop_cc,
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
        nms2d_cuda
    )

# TODO: separate iou and iou loss (the latter one is the diagonal result of previous one)
#       and it seems that we only need the backward part for the iou loss

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

def seg1d_iou(seg1, seg2):
    '''
    Calculate IoU of 1D segments
    The input should be n*2, where the last dim is [center, width]
    '''
    assert torch.all(seg1[:,1] > 0)
    assert torch.all(seg2[:,1] > 0)

    s1max = seg1[:,0] + seg1[:,1] / 2
    s1min = seg1[:,0] - seg1[:,1] / 2
    s2max = seg2[:,0] + seg2[:,1] / 2
    s2min = seg2[:,0] - seg2[:,1] / 2
    
    imax = torch.where(s1max > s2max, s2max, s1max)
    imin = torch.where(s1min < s2min, s2min, s1min)
    umax = torch.where(s1max > s2max, s1max, s2max)
    umin = torch.where(s1min < s2min, s1min, s2min)
    
    i = torch.clamp_min(imax - imin, 0)
    u = torch.clamp_min(umax - umin, 1e-6)
    return i / u

def box2d_iou(boxes1, boxes2, method="box"):
    '''
    :param method: 'box' - normal box, 'rbox' - rotated box,
        'grbox' - giou for rotated box, 'drbox' - diou for rotated box
    '''
    convert_numpy = False
    if isinstance(boxes1, np.ndarray):
        assert isinstance(boxes2, np.ndarray), "Input should be both numpy tensor or pytorch tensor!"
        boxes1 = torch.from_numpy(boxes1)
        boxes2 = torch.from_numpy(boxes2)
        convert_numpy = True

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

    if convert_numpy:
        return result.numpy()
    return result

def box2d_nms(boxes, scores, iou_method="box", supression_method="hard",
    iou_threshold=0, score_threshold=0, supression_param=0):
    '''
    :param method: 'box' - normal box, 'rbox' - rotated box

    Soft-NMS: Bodla, Navaneeth, et al. "Soft-NMS--improving object detection with one line of code." Proceedings of the IEEE international conference on computer vision. 2017.
    '''
    convert_numpy = False
    if isinstance(boxes, np.ndarray):
        assert isinstance(scores, np.ndarray), "Input should be both numpy tensor or pytorch tensor!"
        boxes = torch.from_numpy(boxes)
        scores = torch.from_numpy(scores)
        convert_numpy = True

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

def box2d_crop(cloud, boxes):
    '''
    Crop point cloud points out given rotated boxes.
    The result is a list of indices tensor where each tensor is corresponding to indices of points lying in the box
    '''

    result = rbox_2d_crop_cc(cloud, boxes)
    return result
