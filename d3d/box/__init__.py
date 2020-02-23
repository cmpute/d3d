import torch
from ._impl import rbox_2d_iou as rbox_2d_iou_cc, rbox_2d_iou_cuda, \
    rbox_2d_nms as rbox_2d_nms_cc, rbox_2d_nms_cuda

def rbox_2d_iou(boxes1, boxes2):
    if len(boxes1.shape) != 2 or len(boxes2.shape) != 2:
        raise ValueError("Input of rbox_2d_iou should be Nx2 tensors!")
    if boxes1.shape[1] != 5 or boxes2.shape[1] != 5:
        raise ValueError("Input boxes should have 5 fields: x, y, w, h, r")

    boxes1, boxes2 = boxes1.contiguous(), boxes2.contiguous()

    if boxes1.is_cuda and boxes2.is_cuda:
        ious = torch.empty((len(boxes1), len(boxes2)), dtype=torch.float, device=boxes1.device)
        rbox_2d_iou_cuda(boxes1, boxes2, ious)
    else:
        ious = torch.empty((len(boxes1), len(boxes2)), dtype=torch.float)
        rbox_2d_iou_cc(boxes1, boxes2, ious)
    
    return ious

def rbox_2d_nms(boxes, scores, threshold=0):
    if len(boxes) != len(scores):
        raise ValueError("Numbers of boxes and scores are inconsistent!")
    if len(scores.shape) == 2:
        scores = scores.max(axis=1).values

    if boxes.is_cuda and scores.is_cuda:
        order = scores.argsort(descending=True)
        suppressed = torch.zeros(len(boxes), dtype=bool, device=boxes.device)
        rbox_2d_nms_cuda(boxes, order, threshold, suppressed) 
    else:
        order = scores.argsort(descending=True)
        suppressed = torch.zeros(len(boxes), dtype=bool)
        rbox_2d_nms_cc(boxes, order, threshold, suppressed) 
    return ~suppressed

# TODO: implement softnms
# https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py
