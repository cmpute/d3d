import numpy as np
import shapely.ops as so
import torch
import unittest
from scipy.spatial.distance import cdist
from shapely.geometry import asPolygon

from d3d.geometry import *

eps = 1e-3 # used to avoid unstability

def test_create_line():
    p1, p2 = Point2(1, 2), Point2(3, 4)
    l = line2_from_pp(p1, p2)

    p1, p2 = Point2(1, 2), Point2(1, 1)
    l = line2_from_pp(p1, p2)
    assert l.c == 1

    p1, p2 = Point2(1, 2), Point2(2, 2)
    l = line2_from_pp(p1, p2)
    assert l.c == 2

def test_create_polygon():
    box = poly2_from_xywhr(0, 0, 2, 2, 0.1)
    assert np.isclose(area(box), 4)

def test_intersect():
    # line intersection
    l1 = line2_from_xyxy(0, 0, 1, 1)
    l2 = line2_from_xyxy(0, 1, 1, 0)
    pi = intersect(l1, l2)
    assert np.isclose(pi.x, 0.5) and np.isclose(pi.y, 0.5)

    # bounding box intersection
    b1, b2 = AABox2(1, 3, 1, 3), AABox2(2, 4, 2, 4)
    bi = intersect(b1, b2)
    assert bi.min_x == 2 and bi.min_y == 2
    assert bi.max_x == 3 and bi.max_y == 3

    # polygon intersection
    b1 = poly2_from_xywhr(0, 0, 2, 2, eps)
    b2 = poly2_from_xywhr(0, 0, 2, 2, 1)
    bi = intersect(b1, b2)
    assert 3.14159 < area(bi) < 4

    b1 = poly2_from_xywhr(1, 1, 2, 2, eps)
    b2 = poly2_from_xywhr(2+eps, 2-eps, 2, 2, np.pi/4)
    bi = intersect(b1, b2)
    assert bi.nvertices in [3, 4]
    assert np.isclose(area(bi), 1)

def test_merge():
    # test polygon merge
    b1 = poly2_from_xywhr(0, 0, 4, 2, eps)
    b2 = poly2_from_xywhr(0, 0, 2, 4, -eps)
    bi = merge(b1, b2)
    assert bi.nvertices == 8
    assert area(bi) > 16 - area(intersect(b1, b2))

    b1 = poly2_from_xywhr(-2, 0, 4, 2, eps)
    b2 = poly2_from_xywhr(0, 0, 2, 4, -eps)
    bi = merge(b1, b2)
    assert bi.nvertices == 6
    assert area(bi) > 16 - area(intersect(b1, b2))

    b1 = poly2_from_xywhr(-2, 0, 4, 4, np.pi/4)
    b2 = poly2_from_xywhr(0, 0, 2, 4, eps)
    bi = merge(b1, b2)
    assert bi.nvertices == 5
    assert area(bi) > 24 - area(intersect(b1, b2))

    # test aabox merge
    b1, b2 = AABox2(1, 3, 1, 3), AABox2(2, 4, 2, 4)
    bi = merge(b1, b2)
    assert bi.min_x == 1 and bi.min_y == 1
    assert bi.max_x == 4 and bi.max_y == 4

def test_with_shapely():
    # randomly generate boxes in [-5, 5] range
    n = 1000
    xs = (np.random.rand(n) - 0.5) * 10
    ys = (np.random.rand(n) - 0.5) * 10
    ws = np.random.rand(n) * 5
    hs = np.random.rand(n) * 5
    rs = (np.random.rand(n) - 0.5) * 10
    boxes = [poly2_from_xywhr(x, y, w, h, r) for x,y,w,h,r in zip(xs, ys, ws, hs, rs)]
    shapely_boxes = [asPolygon([(p.x, p.y) for p in box.vertices]) for box in boxes]

    # compare intersection of boxes
    ipoly = [intersect(boxes[i], boxes[i+1]) for i in range(n-1)]
    iarea = np.array([area(p) for p in ipoly])

    shapely_ipoly = [shapely_boxes[i].intersection(shapely_boxes[i+1]) for i in range(n-1)]
    shapely_iarea = np.array([p.area for p in shapely_ipoly])
    assert np.allclose(iarea, shapely_iarea)

    # compare merge of boxes
    mpoly = [merge(boxes[i], boxes[i+1]) for i in range(n-1)]
    marea = np.array([area(p) for p in mpoly])

    shapely_mpoly = [so.cascaded_union([shapely_boxes[i], shapely_boxes[i+1]]).convex_hull for i in range(n-1)]
    shapely_marea = np.array([p.area for p in shapely_mpoly])
    assert np.allclose(marea, shapely_marea)

    # compare max distance calculation
    md = [max_distance(boxes[i], boxes[i+1]) for i in range(n-1)]

    varr = [np.array([(p.x, p.y) for p in box.vertices]) for box in boxes]
    scipy_md = [np.max(cdist(v1, v2)) for v1, v2 in zip(varr[:-1], varr[1:])]
    assert np.allclose(md, scipy_md)

    # compare dimension of the boxes
    dim = np.array([dimension(p) for p in mpoly])

    varr = [np.array([(p.x, p.y) for p in poly.vertices]) for poly in mpoly]
    scipy_dim = [np.max(cdist(v, v)) for v in varr]
    assert np.allclose(dim, scipy_dim)

    # for i in range(n-1):
    #     if not np.isclose(dim[i], scipy_dim[i]):
    #         np.save("b1p.npy", np.array([xs[i], ys[i], ws[i], hs[i], rs[i]]))
    #         np.save("b2p.npy", np.array([xs[i+1], ys[i+1], ws[i+1], hs[i+1], rs[i+1]]))

class TestWithAutograd(unittest.TestCase):
    '''
    This class contains pytorch implementation of some geometry algorithms so that we can
        compare gradient calculation against pytorch autograd.
    '''
    def poly2_from_xywhr(self, data):
        x, y, w, h, r = data
        dxsin, dxcos = w*torch.sin(r)/2, w*torch.cos(r)/2
        dysin, dycos = h*torch.sin(r)/2, h*torch.cos(r)/2

        p0 = torch.stack([x - dxcos + dysin, y - dxsin - dycos])
        p1 = torch.stack([x + dxcos + dysin, y + dxsin - dycos])
        p2 = torch.stack([x + dxcos - dysin, y + dxsin + dycos])
        p3 = torch.stack([x - dxcos - dysin, y - dxsin + dycos])
        return torch.stack([p0, p1, p2, p3])

    def line2_from_pp(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a, b, c = y2-y1, x1-x2, x2*y1-x1*y2
        return torch.stack([a, b, c])

    def intersect_line2(self, l1, l2):
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        w = a1*b2 - a2*b1
        x, y = (b1*c2 - b2*c1) / w, (c1*a2 - c2*a1) / w
        return torch.stack([x, y])

    def intersect_poly2(self, p1, p2, xflags): # reconstruct intersection using xflags
        if len(xflags) == 0:
            return torch.tensor([], dtype=float)

        result = [None] * len(xflags)
        for i in range(len(xflags)):
            iprev = (i-1) % len(xflags)
            if (xflags[i] & 1) == (xflags[iprev] & 1):
                if xflags[i] & 1:
                    result[i] = p1[xflags[i] >> 1]
                else:
                    result[i] = p2[xflags[i] >> 1]
            else:
                if xflags[i] & 1:
                    epni, eppi = xflags[i] >> 1, xflags[iprev] >> 1
                    epnj, eppj = (epni+1) % len(p1), (eppi+1) % len(p2)
                    edge_next = self.line2_from_pp(p1[epni], p1[epnj])
                    edge_prev = self.line2_from_pp(p2[eppi], p2[eppj])
                else:
                    epni, eppi = xflags[i] >> 1, xflags[iprev] >> 1
                    epnj, eppj = (epni+1) % len(p2), (eppi+1) % len(p1)
                    edge_next = self.line2_from_pp(p2[epni], p2[epnj])
                    edge_prev = self.line2_from_pp(p1[eppi], p1[eppj])
                result[i] = self.intersect_line2(edge_prev, edge_next)
        return torch.stack(result)

    def merge_poly2(self, p1, p2, mflags): # reconstruction merged hull using mflags:
        if len(mflags) == 0:
            return torch.tensor([], dtype=float)

        result = [None] * len(mflags)
        for i, m in enumerate(mflags):
            if m & 1:
                result[i] = p1[m >> 1]
            else:
                result[i] = p2[m >> 1]
        return torch.stack(result)

    def area_poly2(self, p):
        if len(p) == 0:
            return torch.tensor(0, dtype=float)

        s = p[-1,0]*p[0,1] - p[-1,1]*p[0,0]
        for i in range(1, len(p)):
            s = s + p[i-1,0]*p[i,1] - p[i,0]*p[i-1,1]
        return s / 2

    def test_iou_grad(self):
        xs = (np.random.rand(2) - 0.5) * 10
        ys = (np.random.rand(2) - 0.5) * 10
        ws = np.random.rand(2) * 5
        hs = np.random.rand(2) * 5
        rs = (np.random.rand(2) - 0.5) * 10

        # polygon intersection
        b1 = poly2_from_xywhr(xs[0], ys[0], ws[0], hs[0], rs[0])
        b2 = poly2_from_xywhr(xs[1], ys[1], ws[1], hs[1], rs[1])
        biou, xflags = iou_(b1, b2)

        grad = np.random.rand()
        grad_p1, grad_p2 = iou_grad(b1, b2, grad, xflags)
        grad1 = poly2_from_xywhr_grad(xs[0], ys[0], ws[0], hs[0], rs[0], grad_p1)
        grad2 = poly2_from_xywhr_grad(xs[1], ys[1], ws[1], hs[1], rs[1], grad_p2)

        # autograd backward calculation
        b1p = torch.tensor([xs[0], ys[0], ws[0], hs[0], rs[0]], dtype=float)
        b2p = torch.tensor([xs[1], ys[1], ws[1], hs[1], rs[1]], dtype=float)
        b1p.requires_grad = True
        b2p.requires_grad = True

        b1v = self.poly2_from_xywhr(b1p)
        b2v = self.poly2_from_xywhr(b2p)
        
        bi = self.intersect_poly2(b1v, b2v, xflags)
        area_i = self.area_poly2(bi)
        area_u = self.area_poly2(b1v) + self.area_poly2(b2v) - area_i
        biou_torch = area_i / area_u
        assert np.isclose(biou, biou_torch.detach())

        biou_torch.backward(torch.tensor(grad))
        assert np.allclose(grad1, b1p.grad.detach())
        assert np.allclose(grad2, b2p.grad.detach())

    def test_giou_grad(self):
        xs = (np.random.rand(2) - 0.5) * 10
        ys = (np.random.rand(2) - 0.5) * 10
        ws = np.random.rand(2) * 5
        hs = np.random.rand(2) * 5
        rs = (np.random.rand(2) - 0.5) * 10

        # polygon intersection
        b1 = poly2_from_xywhr(xs[0], ys[0], ws[0], hs[0], rs[0])
        b2 = poly2_from_xywhr(xs[1], ys[1], ws[1], hs[1], rs[1])
        giou, xflags, mflags = giou_(b1, b2)

        grad = np.random.rand()
        grad_p1, grad_p2 = giou_grad(b1, b2, grad, xflags, mflags)
        grad1 = poly2_from_xywhr_grad(xs[0], ys[0], ws[0], hs[0], rs[0], grad_p1)
        grad2 = poly2_from_xywhr_grad(xs[1], ys[1], ws[1], hs[1], rs[1], grad_p2)

        # autograd backward calculation
        b1p = torch.tensor([xs[0], ys[0], ws[0], hs[0], rs[0]], dtype=float)
        b2p = torch.tensor([xs[1], ys[1], ws[1], hs[1], rs[1]], dtype=float)
        b1p.requires_grad = True
        b2p.requires_grad = True

        b1v = self.poly2_from_xywhr(b1p)
        b2v = self.poly2_from_xywhr(b2p)
        
        bi = self.intersect_poly2(b1v, b2v, xflags)
        bm = self.merge_poly2(b1v, b2v, mflags)
        area_i = self.area_poly2(bi)
        area_m = self.area_poly2(bm)
        area_u = self.area_poly2(b1v) + self.area_poly2(b2v) - area_i
        giou_torch = area_i / area_u + area_u / area_m - 1
        assert np.isclose(giou, giou_torch.detach())

        giou_torch.backward(torch.tensor(grad))
        assert np.allclose(grad1, b1p.grad.detach())
        assert np.allclose(grad2, b2p.grad.detach())

if __name__ == "__main__":
    TestWithAutograd().test_giou_grad()

    # b1p, b2p = np.load("b1p.npy"), np.load("b2p.npy")
    # b1 = poly2_from_xywhr(b1p[0], b1p[1], b1p[2], b1p[3], b1p[4])
    # b2 = poly2_from_xywhr(b2p[0], b2p[1], b2p[2], b2p[3], b2p[4])

    # b1 = poly2_from_xywhr(0, 0, 2, 2, 0)
    # b2 = poly2_from_xywhr(2, 2, 2*np.sqrt(2), 2*np.sqrt(2), np.pi/4)
    # bi = intersect(b1, b2)
    # bm = merge(b1, b2)
    # print([(p.x, p.y) for p in bm.vertices])
    # print(area(bm), bm.nvertices)

    # b1s = asPolygon([(p.x, p.y) for p in b1.vertices])
    # b2s = asPolygon([(p.x, p.y) for p in b2.vertices])
    # bis = b1s.intersection(b2s)
    # bms = so.cascaded_union([b1s, b2s]).convex_hull
    # print(bms.exterior)

    # from matplotlib import pyplot as plt
    # plt.plot(b1s.exterior.xy[0], b1s.exterior.xy[1])
    # plt.plot(b2s.exterior.xy[0], b2s.exterior.xy[1])
    # plt.plot(bms.exterior.xy[0], bms.exterior.xy[1])
    # for i, p in enumerate(b1.vertices): plt.text(p.x, p.y, "1-" + str(i))
    # for i, p in enumerate(b2.vertices): plt.text(p.x, p.y, "2-" + str(i))
    # plt.show()
