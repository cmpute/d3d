import torch
from .box_impl import (
    box_2d_iou as box_2d_iou_cc, box_2d_iou_cuda,
    box_2d_nms as box_2d_nms, box_2d_nms_cuda,
    rbox_2d_iou as rbox_2d_iou_cc, rbox_2d_iou_cuda,
    rbox_2d_nms as rbox_2d_nms_cc, rbox_2d_nms_cuda,
    rbox_2d_crop as rbox_2d_crop_cc)

def box2d_iou(boxes1, boxes2, method="box"):
    '''
    :param method: 'box' - normal box, 'rbox' - rotated box
    '''
    if len(boxes1.shape) != 2 or len(boxes2.shape) != 2:
        raise ValueError("Input of rbox_2d_iou should be Nx2 tensors!")
    if boxes1.shape[1] != 5 or boxes2.shape[1] != 5:
        raise ValueError("Input boxes should have 5 fields: x, y, w, h, r")

    boxes1, boxes2 = boxes1.contiguous(), boxes2.contiguous()

    if boxes1.is_cuda and boxes2.is_cuda:
        if method == "box":
            impl = box_2d_iou_cuda
        elif method == "rbox":
            impl = rbox_2d_iou_cuda
    else:
        if method == "box":
            impl = box_2d_iou_cc
        elif method == "rbox":
            impl = rbox_2d_iou_cc
    return impl(boxes1, boxes2)

def box2d_nms(boxes, scores, method="box", threshold=0):
    '''
    :param method: 'box' - normal box, 'rbox' - rotated box
    '''
    if len(boxes) != len(scores):
        raise ValueError("Numbers of boxes and scores are inconsistent!")
    if len(scores.shape) == 2:
        scores = scores.max(axis=1).values

    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.bool)

    order = scores.argsort(descending=True)
    if boxes.is_cuda and scores.is_cuda:
        suppressed = rbox_2d_nms_cuda(boxes, order, threshold) 
    else:
        suppressed = rbox_2d_nms_cc(boxes, order, threshold)
    return ~suppressed

# TODO: implement softnms
# https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py

def box2d_crop(cloud, boxes):
    '''
    Crop point cloud points out given rotated boxes.
    The result is a list of indices tensor where each tensor is corresponding to indices of points lying in the box
    '''

    result = rbox_2d_crop_cc(cloud, boxes)
    return result
