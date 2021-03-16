#pragma once

#include <dgal/geometry.hpp>
#include "d3d/common.h"

bool box3dr_contains(float x, float y, float z, float lx, float ly, float lz, float rz, float xq, float yq, float zq)
{
    dgal::Quad2<float> box = dgal::poly2_from_xywhr(x, y, lx, ly, rz);
    dgal::AABox2<float> aabox = dgal::aabox2_from_poly2(box);
    dgal::Point2<float> p {.x=xq, .y=yq};

    if (zq > z + lz/2 || zq < z - lz/2)
        return false;
    if (!aabox.contains(p))
        return false;
    if (!box.contains(p))
        return false;
    return true;
}

float box2dr_iou(float x1, float y1, float lx1, float ly1, float rz1,
                float x2, float y2, float lx2, float ly2, float rz2)
{
    dgal::Quad2<float> box1 = dgal::poly2_from_xywhr(x1, y1, lx1, ly1, rz1),
                       box2 = dgal::poly2_from_xywhr(x2, y2, lx2, ly2, rz2);
    return dgal::iou(box1, box2);
}

float box2d_iou(float x1, float y1, float lx1, float ly1, float rz1,
                float x2, float y2, float lx2, float ly2, float rz2)
{
    dgal::Quad2<float> box1 = dgal::poly2_from_xywhr(x1, y1, lx1, ly1, rz1),
                       box2 = dgal::poly2_from_xywhr(x2, y2, lx2, ly2, rz2);
    dgal::AABox2<float> aabox1 = dgal::aabox2_from_poly2(box1),
                        aabox2 = dgal::aabox2_from_poly2(box2);
    return dgal::iou(aabox1, aabox2);
}