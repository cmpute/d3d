import torch
import numpy as np
import time
import sys
from collections import defaultdict, namedtuple

# Official inplementation of 3D IoU Loss
# please request the official implementation from the authors
from gious import ious_3D, gious_3D

# 3D IoU Loss implemented by lilanxiao
# https://github.com/lilanxiao/Rotated_IoU
sys.path.append("/home/jacob/PointCloud/d3d/test/compare/Rotated_IoU")
from oriented_iou_loss import cal_iou

# RIoU implemented in RRPN
# https://github.com/hongzhenwang/RRPN-revise/blob/master/lib/rotation/test_IoU.py
sys.path.append("/home/jacob/PointCloud/d3d/test/compare/RRPN-revise/lib")
from rotation.rbbox_overlaps import rbbx_overlaps

# RIoU implemented in OpenPCDet
# https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/iou3d_nms_utils.py
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev 

# Our implementation of rotated IoU Loss in d3d
from d3d.box import box2d_iou # our implementations

torch.autograd.set_detect_anomaly(True)


def test_simple():
    boxes1 = torch.tensor([[0, 0, 0, 2, 2, 2, 0.2], [0, 0, 0, 2, 2, 2, 0.4]], requires_grad=False, dtype=torch.float32)
    boxes2 = torch.tensor([[0, 0, 0, 2, 2, 2, 1  ], [0, 0, 0, 2, 2, 2, 1  ]], requires_grad=True, dtype=torch.float32)

    op = ious_3D()
    result = op(boxes1, boxes2, aligned=False)
    print(result)
    loss = result.log().sum()
    loss.backward()
    print(result)

    op = box2d_iou
    result = op(boxes1[:, [0,1,3,4,6]], boxes2[:, [0,1,3,4,6]], method="rbox")
    print(result)


def test_bulk(verbose=False):
    t_result = namedtuple("Record", ["n", "type", "source", "dir"])
    results = defaultdict(list)
    
    warmed_up = False
    tstart = time.time()
    for n, repeat in [
        (1, 100), # fisrt round is used for warm-up
        (1, 1000),
        (2,  500),
        (5,  200),
        (10, 100),
        (20,  50),
        (50,  20),
        (100, 10),
        (200, 10),
        (500,  5),
        (1000, 5),
        (2000, 5),
        (5000, 5)
    ]:
        print("Running n=%d for %d times" % (n, repeat))
        for _ in range(repeat):
            xs = (np.random.rand(n) - 0.5) * 10
            ys = (np.random.rand(n) - 0.5) * 10
            ws = np.random.rand(n) * 5
            hs = np.random.rand(n) * 5
            rs = (np.random.rand(n) - 0.5) * 10
            boxes1 = torch.tensor([[x, y, 0, w, h, 2, r] for x, y, w, h, r in zip(xs, ys, ws, hs, rs)], dtype=torch.float32)

            xs = (np.random.rand(n) - 0.5) * 10
            ys = (np.random.rand(n) - 0.5) * 10
            ws = np.random.rand(n) * 5
            hs = np.random.rand(n) * 5
            rs = (np.random.rand(n) - 0.5) * 10
            boxes2 = torch.tensor([[x, y, 0, w, h, 2, r] for x, y, w, h, r in zip(xs, ys, ws, hs, rs)], dtype=torch.float32, requires_grad=True)

            boxes1_cuda, boxes2_cuda = boxes1.cuda(), boxes2.cuda()
            if verbose: print("==========")

            ### pcdet results ###
            op = lambda b1, b2: boxes_iou_bev(b1, b2)
            # no cpu support

            t1 = time.time()
            result7 = op(boxes1_cuda, boxes2_cuda)
            t2 = time.time()
            if verbose: print("pcdet GPU time %.4fs, output shape %s" % (t2 - t1, result7.shape))
            if warmed_up: results[t_result(n=n, type="gpu", source="pcdet", dir="forward")].append(t2-t1)

            ### Our results ###
            op = lambda b1, b2: box2d_iou(b1[:, [0,1,3,4,6]], b2[:, [0,1,3,4,6]], method="rbox")
            t1 = time.time()
            result1 = op(boxes1, boxes2)
            t2 = time.time()
            if verbose: print("Our CPU time %.4fs, output shape %s" % (t2 - t1, result1.shape))
            if warmed_up: results[t_result(n=n, type="cpu", source="ours", dir="forward")].append(t2-t1)

            loss = result1.sum()
            t1 = time.time()
            loss.backward()
            t2 = time.time()
            if warmed_up: results[t_result(n=n, type="cpu", source="ours", dir="backward")].append(t2-t1)

            t1 = time.time()
            result2 = op(boxes1_cuda, boxes2_cuda)
            t2 = time.time()
            if verbose: print("Our GPU time %.4fs, output shape %s" % (t2 - t1, result2.shape))
            if warmed_up: results[t_result(n=n, type="gpu", source="ours", dir="forward")].append(t2-t1)
            assert torch.allclose(result1, result2.cpu(), atol=1e-6)

            loss = result2.sum()
            t1 = time.time()
            loss.backward()
            t2 = time.time()
            if warmed_up: results[t_result(n=n, type="gpu", source="ours", dir="backward")].append(t2-t1)

            ## rrpn results ###
            op = lambda b1, b2: rbbx_overlaps(b1[:, [0,1,3,4,6]], b2[:, [0,1,3,4,6]])
            # no cuda support

            t1 = time.time()
            result6 = op(boxes1.detach().numpy(), boxes2.detach().numpy())
            t2 = time.time()
            if verbose: print("rrpn CPU time %.4fs, output shape %s" % (t2 - t1, result6.shape))
            if warmed_up: results[t_result(n=n, type="cpu", source="rrpn", dir="forward")].append(t2-t1)

            ############ Other methods are calculating direct iou instead of cross ious ##############
            xx, yy = torch.meshgrid(torch.arange(n), torch.arange(n))
            boxes1 = boxes1[xx.reshape(-1), :]
            boxes2 = boxes2.detach()[yy.reshape(-1), :]
            boxes2.requires_grad = True
            boxes1_cuda, boxes2_cuda = boxes1.cuda(), boxes2.cuda()

            ### Official results ###
            if n < 100:
                op = ious_3D()
                t1 = time.time()
                result3 = op(boxes1, boxes2, aligned=False)
                t2 = time.time()
                if verbose: print("Official CPU time %.4fs, output shape %s" % (t2 - t1, result3.shape))
                if warmed_up: results[t_result(n=n, type="cpu", source="official", dir="forward")].append(t2-t1)

                loss = result3.sum()
                t1 = time.time()
                loss.backward()
                t2 = time.time()
                if warmed_up: results[t_result(n=n, type="cpu", source="official", dir="backward")].append(t2-t1)

                t1 = time.time()
                result4 = op(boxes1_cuda, boxes2_cuda, aligned=False)
                t2 = time.time()
                if verbose: print("Official GPU time %.4fs, output shape %s" % (t2 - t1, result4.shape))
                if warmed_up: results[t_result(n=n, type="gpu", source="official", dir="forward")].append(t2-t1)

                loss = result4.sum()
                t1 = time.time()
                loss.backward()
                t2 = time.time()
                if warmed_up: results[t_result(n=n, type="gpu", source="official", dir="backward")].append(t2-t1)


            ### lilan results ###
            op = lambda b1, b2: cal_iou(b1[:, [0,1,3,4,6]].unsqueeze(0), b2[:, [0,1,3,4,6]].unsqueeze(0))
            # no cpu support

            if n <= 1000:
                t1 = time.time()
                result5, *_ = op(boxes1_cuda, boxes2_cuda)
                t2 = time.time()
                if verbose: print("lilan GPU time %.4fs, output shape %s" % (t2 - t1, result5.shape))
                if warmed_up: results[t_result(n=n, type="gpu", source="lilan", dir="forward")].append(t2-t1)

                loss = result5.sum()
                t1 = time.time()
                loss.backward()
                t2 = time.time()
                if warmed_up: results[t_result(n=n, type="gpu", source="lilan", dir="backward")].append(t2-t1)

        if not warmed_up:
            warmed_up = True

    tend = time.time()
    print("Total time:", tend - tstart)

    # save results
    rsave = {}
    for k, v in results.items():
        arr = np.array(v)
        name = k.source + '.' + k.type + '.' + k.dir + "." + str(k.n)
        rsave[name] = arr
    np.savez("results.npz", **rsave)
    print("Results saved.")

if __name__ == "__main__":
    test_bulk(False)
