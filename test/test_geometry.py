from d3d.geometry import *
import numpy as np

def test_create_line():
    p1, p2 = Point2f(1, 2), Point2f(3, 4)
    l = line2_from_pp(p1, p2)

    p1, p2 = Point2f(1, 2), Point2f(1, 1)
    l = line2_from_pp(p1, p2)
    assert l.c == 1

    p1, p2 = Point2f(1, 2), Point2f(2, 2)
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
    b1, b2 = AABox2f(1, 3, 1, 3), AABox2f(2, 4, 2, 4)
    bi = intersect(b1, b2)
    assert bi.min_x == 2 and bi.min_y == 2
    assert bi.max_x == 3 and bi.max_y == 3

    # polygon intersection
    b1 = poly2_from_xywhr(0, 0, 2, 2, 0)
    b2 = poly2_from_xywhr(0, 0, 2, 2, 1)
    bi = intersect(b1, b2)
    assert 3.14159 < area(bi) < 4

    b1 = poly2_from_xywhr(1, 1, 2, 2, 0)
    b2 = poly2_from_xywhr(2, 2, 2, 2, np.pi/4)
    bi = intersect(b1, b2)
    assert bi.nvertices in [3, 4]
    assert np.isclose(area(bi), 1)

def test_merge():
    # test polygon merge
    b1 = poly2_from_xywhr(0, 0, 4, 2, 0.01)
    b2 = poly2_from_xywhr(0, 0, 2, 4, 0.02)
    bi = merge(b1, b2)
    assert bi.nvertices == 8
    assert area(bi) > 16 - area(intersect(b1, b2))

    b1 = poly2_from_xywhr(-2, 0, 4, 2, 0.01)
    b2 = poly2_from_xywhr(0, 0, 2, 4, 0.02)
    bi = merge(b1, b2)
    assert bi.nvertices == 6
    assert area(bi) > 16 - area(intersect(b1, b2))

    b1 = poly2_from_xywhr(-2, 0, 4, 4, np.pi/4)
    b2 = poly2_from_xywhr(0, 0, 2, 4, 0.02)
    bi = merge(b1, b2)
    assert bi.nvertices == 5
    assert area(bi) > 24 - area(intersect(b1, b2))

    # test aabox merge
    b1, b2 = AABox2f(1, 3, 1, 3), AABox2f(2, 4, 2, 4)
    bi = merge(b1, b2)
    assert bi.min_x == 1 and bi.min_y == 1
    assert bi.max_x == 4 and bi.max_y == 4

if __name__ == "__main__":
    test_merge()
