import torch
from ._impl import rbox_2d_iou as rbox_2d_iou_cc

def rbox_2d_iou(boxes1, boxes2):
    if len(boxes1.shape) != 2 or len(boxes2.shape) != 2:
        raise ValueError("Input of rbox_2d_iou should be Nx2 tensors!")
    if boxes1.shape[1] != 5 or boxes2.shape[1] != 5:
        raise ValueError("Input boxes should have 5 fields: x, y, w, h, r")

    ious = torch.empty((len(boxes1), len(boxes2)), dtype=torch.float)
    rbox_2d_iou_cc(boxes1, boxes2, ious)
    
    return ious

# References:
# https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/nms
# https://github.com/gdlg/pytorch_nms

# def nms(boxes, scores, overlap, top_k):
#     return _impl.nms_forward(boxes, scores, overlap, top_k)
