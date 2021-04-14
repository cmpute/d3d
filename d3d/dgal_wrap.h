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

float box3dr_pdist(float x, float y, float z, float lx, float ly, float lz, float rz, float xq, float yq, float zq)
{
    dgal::Quad2<float> box = dgal::poly2_from_xywhr(x, y, lx, ly, rz);
    dgal::Point2<float> p {.x=xq, .y=yq};

    float dist2d = distance(box, p);
    float distz;
    if (zq > z)
        distz = z + lz / 2 - zq;
    else
        distz = zq - z + lz / 2;

    if (distz > 0) 
        if (dist2d > 0)
            return dgal::_min(distz, dist2d);
        else
            return dist2d;
    else
        if (dist2d > 0)
            return distz;
        else
            return -hypot(distz, dist2d);
}

float box3dr_iou(float x1, float y1, float z1, float lx1, float ly1, float lz1, float rz1,
                 float x2, float y2, float z2, float lx2, float ly2, float lz2, float rz2)
{
    dgal::Quad2<float> box1 = dgal::poly2_from_xywhr(x1, y1, lx1, ly1, rz1),
                       box2 = dgal::poly2_from_xywhr(x2, y2, lx2, ly2, rz2);
    float iou2d = dgal::iou(box1, box2);

    float z1max = z1 + lz1 / 2;
    float z1min = z1 - lz1 / 2;
    float z2max = z2 + lz2 / 2;
    float z2min = z2 - lz2 / 2;

    float imax = dgal::_min(z1max, z2max);
    float imin = dgal::_max(z1min, z2min);
    float umax = dgal::_max(z1max, z2max);
    float umin = dgal::_min(z1min, z2min);
    
    float i = dgal::_max(imax - imin, 0.f);
    float u = dgal::_max(umax - umin, float(1e-6));
    float ziou = i / u;

    return iou2d * ziou;
}
