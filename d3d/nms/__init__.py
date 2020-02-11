from . import _impl

# References:
# https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/nms
# https://github.com/gdlg/pytorch_nms

def nms(boxes, scores, overlap, top_k):
    return _impl.nms_forward(boxes, scores, overlap, top_k)
