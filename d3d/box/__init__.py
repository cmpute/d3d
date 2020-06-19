import torch
from .box_impl import (
    iou2d as iou2d_cc, iou2d_cuda,
    nms2d as nms2d_cc, nms2d_cuda,
    rbox_2d_crop as rbox_2d_crop_cc,
    IouType, SupressionType)

def box2d_iou(boxes1, boxes2, method="box"):
    '''
    :param method: 'box' - normal box, 'rbox' - rotated box
    '''
    if len(boxes1.shape) != 2 or len(boxes2.shape) != 2:
        raise ValueError("Input of rbox_2d_iou should be Nx2 tensors!")
    if boxes1.shape[1] != 5 or boxes2.shape[1] != 5:
        raise ValueError("Input boxes should have 5 fields: x, y, w, h, r")

    iou_type = getattr(IouType, method.upper())
    if boxes1.is_cuda and boxes2.is_cuda:
        impl = iou2d_cuda
    else:
        impl = iou2d_cc
    return impl(boxes1, boxes2, iou_type)

# TODO: implement IoU loss, GIoU, DIoU, CIoU: https://zhuanlan.zhihu.com/p/104236411

def box2d_nms(boxes, scores, iou_method="box", supression_method="hard",
    iou_threshold=0, score_threshold=0, supression_param=0):
    '''
    :param method: 'box' - normal box, 'rbox' - rotated box

    Soft-NMS: Bodla, Navaneeth, et al. "Soft-NMS--improving object detection with one line of code." Proceedings of the IEEE international conference on computer vision. 2017.
    '''
    if len(boxes) != len(scores):
        raise ValueError("Numbers of boxes and scores are inconsistent!")
    if len(scores.shape) == 2:
        scores = scores.max(axis=1).values

    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.bool)

    iou_type = getattr(IouType, iou_method.upper())
    supression_type = getattr(SupressionType, supression_method.upper())

    if boxes.is_cuda and scores.is_cuda:
        impl = nms2d_cuda
    else:
        impl = nms2d_cc

    suppressed = impl(boxes, scores,
        iou_type, supression_type,
        iou_threshold, score_threshold, supression_param
    )
    return ~suppressed

def box2d_crop(cloud, boxes):
    '''
    Crop point cloud points out given rotated boxes.
    The result is a list of indices tensor where each tensor is corresponding to indices of points lying in the box
    '''

    result = rbox_2d_crop_cc(cloud, boxes)
    return result
