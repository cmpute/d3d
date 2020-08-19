/*
 * Copyright (c) 2019- Yuanxin Zhong. All rights reserved.
 * This file contains implementations of algorithms of simple geometries suitable for using in GPU.
 * One can turn to shapely, boost.geometry or CGAL for a complete functionality of geometry operations.
 * 
 * Predicates:
 *      .contains: whether the shape contains another shape
 *      .intersects: whether the shape has intersection with another shaoe
 * 
 * Unary Functions:
 *      .area: calculate the area of the shape
 *      .dimension: calculate the largest distance between any two vertices
 * 
 * Binary Operations:
 *      .distance: calculate the (minimum) distance between two shapes
 *      .max_distance: calculate the (maximum) distance between two shapes.
 *      .intersect: calculate the intersection of the two shapes
 *      .merge: calculate the shape with minimum area that contains the two shapes
 *
 * Extension Operations:
 *      .iou: calculate the intersection over union (about area)
 *      .giou: calculate the generalized intersection over union
 *          (ref "Generalized Intersection over Union")
 *      .diou: calculate the distance intersection over union
 *          (ref "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression")
 * 
 * Note that the library haven't been tested for stability, parallel edge and vertex on edge
 * should be avoided! This is due to both the complex implementation and incompatible gradient calculation.
 */


#ifndef D3D_GEOMETRY_HPP
#define D3D_GEOMETRY_HPP

#include <cmath>
#include <limits>
#include "d3d/common.h"

namespace d3d {

//////////////////// forward declarations //////////////////////
template <typename scalar_t> struct Point2;
template <typename scalar_t> struct Line2;
template <typename scalar_t> struct AABox2;
template <typename scalar_t, uint8_t MaxPoints> struct Poly2;
template <typename scalar_t> using Box2 = Poly2<scalar_t, 4>; // TODO: rename as Quad4

using Point2f = Point2<float>;
using Point2d = Point2<double>;
using Line2f = Line2<float>;
using Line2d = Line2<double>;
using AABox2f = AABox2<float>;
using AABox2d = AABox2<double>;
template <uint8_t MaxPoints> using Poly2f = Poly2<float, MaxPoints>;
template <uint8_t MaxPoints> using Poly2d = Poly2<double, MaxPoints>;
using Box2f = Box2<float>;
using Box2d = Box2<double>;

////////////////////////// Helpers //////////////////////////
#ifndef GEOMETRY_EPS
    constexpr double _eps = 1e-6;
#else
    constexpr double _eps = GEOMETRY_EPS;
#endif
constexpr double _pi = 3.14159265358979323846;

template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _max(const T &a, const T &b) { return a > b ? a : b; }
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _min(const T &a, const T &b) { return a < b ? a : b; }
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _mod_inc(const T &i, const T &n) { return i < n - 1 ? (i + 1) : 0; }
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _mod_dec(const T &i, const T &n) { return i > 0 ? (i - 1) : (n - 1); }

///////////////////// implementations /////////////////////
template <typename scalar_t> struct Point2 // Point in 2D surface
{
    scalar_t x, y;
};

// Calculate cross product (or area) of vector p1->p2 and p2->t
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t _cross(const Point2<scalar_t> &p1, const Point2<scalar_t> &p2, const Point2<scalar_t> &t)
{
    return (p2.x - p1.x) * (t.y - p2.y) - (p2.y - p1.y) * (t.x - p2.x);
}

template <typename scalar_t> struct Line2 // Infinite but directional line
{
    scalar_t a, b, c; // a*x + b*y + c = 0

    CUDA_CALLABLE_MEMBER inline bool intersects(const Line2<scalar_t> &l) const
    {
        return a*l.b - l.a*b == 0;
    }
};

template <typename scalar_t> struct AABox2 // Axis-aligned 2D box for quick calculation
{
    scalar_t min_x, max_x, min_y, max_y;

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2<scalar_t>& p) const
    {
        return p.x > min_x && p.x < max_x && p.y > min_y && p.y < max_y;
    }

    CUDA_CALLABLE_MEMBER inline bool contains(const AABox2<scalar_t>& a) const
    {
        return max_x > a.max_x && min_x < a.min_x && max_y > a.max_y && min_y < a.min_y;
    }

    CUDA_CALLABLE_MEMBER inline bool intersects(const AABox2<scalar_t>& a) const
    {
        return max_x > a.min_x && min_x < a.max_x && max_y > a.min_y && min_y < a.max_y;
    }
};

template <typename scalar_t, uint8_t MaxPoints> struct Poly2 // Convex polygon with no holes
{
    Point2<scalar_t> vertices[MaxPoints]; // vertices in counter-clockwise order
    uint8_t nvertices = 0; // actual number of vertices, max support 128 vertices right now

    template <uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER
    Poly2<scalar_t, MaxPoints>& operator=(Poly2<scalar_t, MaxPoints2> const &other)
    {
        assert(other.nvertices <= MaxPoints);
        nvertices = other.nvertices;
        for (uint8_t i = 0; i < other.nvertices; i++)
            vertices[i] = other.vertices[i];
        return *this;
    }

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2<scalar_t>& p) const
    { 
        // deal with head and tail first
        if (_cross(vertices[nvertices-1], vertices[0], p) < 0) return false;

        // then loop remaining edges
        for (uint8_t i = 1; i < nvertices; i++)
            if (_cross(vertices[i-1], vertices[i], p) < 0)
                return false;

        return true;
    }
};

////////////////// constructors ///////////////////

// Note that the order of points matters
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
Line2<scalar_t> line2_from_xyxy(const scalar_t &x1, const scalar_t &y1,
    const scalar_t &x2, const scalar_t &y2)
{
    return {.a=y2-y1, .b=x1-x2, .c=x2*y1-x1*y2};
}

// Note that the order of points matters
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
Line2<scalar_t> line2_from_pp(Point2<scalar_t> p1, Point2<scalar_t> p2)
{
    return line2_from_xyxy(p1.x, p1.y, p2.x, p2.y);
}

// convert AABox2 to a polygon (box)
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, 4> poly2_from_aabox2(const AABox2<scalar_t> &a)
{
    Point2<scalar_t> p1{a.min_x, a.min_y},
        p2{a.max_x, a.min_y},
        p3{a.max_x, a.max_y},
        p4{a.min_x, a.max_y};
    return {.vertices={p1, p2, p3, p4}, .nvertices=4};
}

// calculate bounding box of a polygon
template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
AABox2<scalar_t> aabox2_from_poly2(const Poly2<scalar_t, MaxPoints> &p)
{
    AABox2<scalar_t> result {
        .min_x = p.vertices[0].x, .max_x = p.vertices[0].x,
        .min_y = p.vertices[0].y, .max_y = p.vertices[0].y
    };

    for (uint8_t i = 1; i < p.nvertices; i++)
    {
        result.min_x = _min(p.vertices[i].x, result.min_x);
        result.max_x = _max(p.vertices[i].x, result.max_x);
        result.min_y = _min(p.vertices[i].y, result.min_y);
        result.max_y = _max(p.vertices[i].y, result.max_y);
    }
    return result;
}

// Create a polygon from box parameters (x, y, width, height, rotation)
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, 4> poly2_from_xywhr(const scalar_t& x, const scalar_t& y,
    const scalar_t& w, const scalar_t& h, const scalar_t& r)
{
    scalar_t dxsin = w*sin(r)/2, dxcos = w*cos(r)/2;
    scalar_t dysin = h*sin(r)/2, dycos = h*cos(r)/2;

    Point2<scalar_t> p0 {.x = x - dxcos + dysin, .y = y - dxsin - dycos};
    Point2<scalar_t> p1 {.x = x + dxcos + dysin, .y = y + dxsin - dycos};
    Point2<scalar_t> p2 {.x = x + dxcos - dysin, .y = y + dxsin + dycos};
    Point2<scalar_t> p3 {.x = x - dxcos - dysin, .y = y - dxsin + dycos};
    return {.vertices={p0, p1, p2, p3}, .nvertices=4};
}

//////////////////// functions ///////////////////

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t distance(const Point2<scalar_t> &p1, const Point2<scalar_t> &p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// Calculate signed distance from point p to the line
// The distance is negative if the point is at left hand side wrt the direction of line (x1y1 -> x2y2)
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t distance(const Line2<scalar_t> &l, const Point2<scalar_t> &p)
{
    return (l.a*p.x + l.b*p.y + l.c) / sqrt(l.a*l.a + l.b*l.b);
}

// Calculate intersection point of two lines
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
Point2<scalar_t> intersect(const Line2<scalar_t> &l1, const Line2<scalar_t> &l2)
{
    scalar_t w = l1.a*l2.b - l2.a*l1.b;
    return {.x=(l1.b*l2.c - l2.b*l1.c) / w, .y=(l1.c*l2.a - l2.c*l1.a) / w};
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
AABox2<scalar_t> intersect(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2)
{
    if (a1.max_x <= a2.min_x || a1.min_x >= a2.max_x ||
        a1.max_y <= a2.min_y || a1.min_y >= a2.max_y)
        // return empty aabox if no intersection
        return {0, 0, 0, 0};
    else
    {
        return {
            .min_x = _max(a1.min_x, a2.min_x),
            .max_x = _min(a1.max_x, a2.max_x),
            .min_y = _max(a1.min_y, a2.min_y),
            .max_y = _min(a1.max_y, a2.max_y)
        };
    }
}

// Find the intersection point under a bridge over two convex polygons
// p1 is searched from idx1 in clockwise order and p2 is searched from idx2 in counter-clockwise order
// the indices of edges of the intersection are reported by xidx1 and xidx2
// note: index of an edge is the index of its (counter-clockwise) starting point
// If intersection is not found, then false will be returned
template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
bool _find_intersection_under_bridge(const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    const uint8_t &idx1, const uint8_t &idx2, uint8_t &xidx1, uint8_t &xidx2)
{
    uint8_t i1 = idx1, i2 = idx2;
    bool finished = false;
    scalar_t dist, last_dist1, last_dist2;
    last_dist1 = last_dist2 = std::numeric_limits<scalar_t>::infinity();
    Line2<scalar_t> edge;

    while (!finished)
    { 
        finished = true;

        // traverse down along p2
        edge = line2_from_pp(p1.vertices[_mod_dec(i1, p1.nvertices)], p1.vertices[i1]);
        while (true)
        {
            dist = distance(edge, p2.vertices[_mod_inc(i2, p2.nvertices)]);
            if (dist > last_dist1) return false;
            if (dist < 0) break;

            i2 = _mod_inc(i2, p2.nvertices);
            last_dist1 = dist;
            finished = false;
        }

        // traverse down along p1
        edge = line2_from_pp(p2.vertices[i2], p2.vertices[_mod_inc(i2, p2.nvertices)]);
        while (true)
        {
            dist = distance(edge, p1.vertices[_mod_dec(i1, p1.nvertices)]);
            if (dist > last_dist2) return false;
            if (dist < 0) break;

            i1 = _mod_dec(i1, p2.nvertices);
            last_dist2 = dist;
            finished = false;
        }
    }

    // TODO: maybe need to test wether the last two segments actually intersect
    xidx1 = _mod_dec(i1, p1.nvertices);
    xidx2 = i2;
    return true;
}

// Find the extreme vertex in a polygon
// TopRight=true is finding point with biggest y
// TopRight=false if finding point with smallest y
template <typename scalar_t, uint8_t MaxPoints, bool TopRight=true> CUDA_CALLABLE_MEMBER inline
void _find_extreme(const Poly2<scalar_t, MaxPoints> &p, uint8_t &idx, scalar_t &ey)
{
    idx = 0;
    ey = p.vertices[0].y;
    for (uint8_t i = 1; i < p.nvertices; i++)
        if ((TopRight && p.vertices[i].y > ey) || (!TopRight && p.vertices[i].y < ey))
        {
            ey = p.vertices[i].y;
            idx = i;
        }
}

// Calculate slope angle from p1 to p2
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t _slope_pp(const Point2<scalar_t> &p1, const Point2<scalar_t> &p2)
{
    return atan2(p2.y - p1.y - _eps, p2.x - p1.x); // eps is used to let atan2(0, -1)=-pi
}

// check if the bridge defined between two polygon (from p1 to p2) is valid.
// reverse is set to true if the polygons lay in the right of the bridge
template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
bool _check_valid_bridge(const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    const uint8_t &idx1, const uint8_t &idx2, bool &reverse)
{
    scalar_t d1 = _cross(p1.vertices[idx1], p2.vertices[idx2], p1.vertices[_mod_dec(idx1, p1.nvertices)]);
    scalar_t d2 = _cross(p1.vertices[idx1], p2.vertices[idx2], p1.vertices[_mod_inc(idx1, p1.nvertices)]);
    scalar_t d3 = _cross(p1.vertices[idx1], p2.vertices[idx2], p2.vertices[_mod_dec(idx2, p2.nvertices)]);
    scalar_t d4 = _cross(p1.vertices[idx1], p2.vertices[idx2], p2.vertices[_mod_inc(idx2, p2.nvertices)]);

    reverse = d1 < 0;
    return d1*d2 > 0 && d2*d3 > 0 && d3*d4 > 0;
}

// Rotating Caliper implementation of intersecting
// Reference: https://web.archive.org/web/20150415231115/http://cgm.cs.mcgill.ca/~orm/rotcal.frame.htm
//    and http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/97/Plante/CompGeomProject-EPlante/algorithm.html
// xflags stores edges flags for gradient computation:
// left 7bits from the 8bits are index number of the edge (index of the first vertex of edge)
// right 1 bit indicate whether edge is from p1 (=1) or p2 (=0)
// the vertices of the output polygon are ordered so that vertex i is the intersection of edge i-1 and edge i in the flags
template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, MaxPoints1 + MaxPoints2> intersect(
    const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    uint8_t xflags[MaxPoints1 + MaxPoints2] = nullptr
) {
    // find the vertices with max y value, starting from line pointing to -x (angle is -pi)
    uint8_t pidx1, pidx2; scalar_t _;
    _find_extreme(p1, pidx1, _);
    _find_extreme(p2, pidx2, _);

    // start rotating to find all intersection points
    scalar_t edge_angle = -_pi; // scan from -pi to pi
    bool edge_flag; // true: intersection connection will start with p1, false: start with p2
    bool reverse; // temporary var
    uint8_t nx = 0; // number of intersection points
    uint8_t x1indices[MaxPoints1 + MaxPoints2 + 1], x2indices[MaxPoints1 + MaxPoints2 + 1];

    while (true)
    {
        uint8_t pidx1_next = _mod_inc(pidx1, p1.nvertices);
        uint8_t pidx2_next = _mod_inc(pidx2, p2.nvertices);
        scalar_t angle1 = _slope_pp(p1.vertices[pidx1], p1.vertices[pidx1_next]);
        scalar_t angle2 = _slope_pp(p2.vertices[pidx2], p2.vertices[pidx2_next]);

        // compare angles and proceed
        if (edge_angle < angle1 && (angle1 < angle2 || angle2 < edge_angle))
        { // choose p1 edge as part of co-podal pair

            if (_check_valid_bridge(p1, p2, pidx1_next, pidx2, reverse)) // valid bridge
            {
                bool has_intersection = reverse ?
                    _find_intersection_under_bridge(p1, p2, pidx1_next, pidx2, x1indices[nx], x2indices[nx]):
                    _find_intersection_under_bridge(p2, p1, pidx2, pidx1_next, x2indices[nx], x1indices[nx]);
                if (!has_intersection)
                    return {}; // return empty polygon
                
                // save intersection
                if (nx == 0) edge_flag = !reverse;
                nx++;
            }

            // update pointer
            pidx1 = _mod_inc(pidx1, p1.nvertices);
            edge_angle = angle1;
        }
        else if (edge_angle < angle2 && (angle2 < angle1 || angle1 < edge_angle))
        { // choose p2 edge as part of co-podal pair

            if (_check_valid_bridge(p1, p2, pidx1, pidx2_next, reverse)) // valid bridge
            {
                bool has_intersection = reverse ?
                    _find_intersection_under_bridge(p1, p2, pidx1, pidx2_next, x1indices[nx], x2indices[nx]):
                    _find_intersection_under_bridge(p2, p1, pidx2_next, pidx1, x2indices[nx], x1indices[nx]);
                if (!has_intersection)
                    return {}; // return empty polygon
                
                // save intersection
                if (nx == 0) edge_flag = !reverse;
                nx++;
            }

            // update pointer
            pidx2 = _mod_inc(pidx2, p2.nvertices);
            edge_angle = angle2;
        }
        else break; // when both angles are not increasing, the loop is finished
    }

    // if no intersection found but didn't return early, then containment is detected
    Poly2<scalar_t, MaxPoints1 + MaxPoints2> result;
    if (nx == 0)
    {
        if (area(p1) > area(p2))
        {
            result = p2;
            if (xflags != nullptr)
                for (uint8_t i = 0; i < p2.nvertices; i++)
                    xflags[i] = i << 1;
        }
        else
        {
            result = p1;
            if (xflags != nullptr)
                for (uint8_t i = 0; i < p1.nvertices; i++)
                    xflags[i] = (i << 1) | 1;
        }
        return result;
    }

    // loop over the intersections to construct the result polygon
    x1indices[nx] = x1indices[0]; // add sentinels
    x2indices[nx] = x2indices[0];
    for (uint8_t i = 0; i < nx; i++)
    {
        // add intersection point
        Line2<scalar_t> l1 = line2_from_pp(p1.vertices[x1indices[i]], p1.vertices[_mod_inc(x1indices[i], p1.nvertices)]);
        Line2<scalar_t> l2 = line2_from_pp(p2.vertices[x2indices[i]], p2.vertices[_mod_inc(x2indices[i], p2.nvertices)]);
        if (xflags != nullptr)
            xflags[result.nvertices] = edge_flag ? (x1indices[i] << 1 | 1) : (x2indices[i] << 1);
        result.vertices[result.nvertices++] = intersect(l1, l2);

        // add points between intersections
        if (edge_flag)
        {
            uint8_t idx_next = x1indices[i+1] >= x1indices[i] ? x1indices[i+1] : (x1indices[i+1] + p1.nvertices);
            for (uint8_t j = x1indices[i] + 1; j <= idx_next; j++)
            {
                uint8_t jmod = j < p1.nvertices ? j : (j - p1.nvertices);
                if (xflags != nullptr)
                    xflags[result.nvertices] = (jmod << 1) | 1;
                result.vertices[result.nvertices++] = p1.vertices[jmod];
            }
        }
        else
        {
            uint8_t idx_next = x2indices[i+1] >= x2indices[i] ? x2indices[i+1] : (x2indices[i+1] + p2.nvertices);
            for (uint8_t j = x2indices[i] + 1; j <= idx_next; j++)
            {
                uint8_t jmod = j < p2.nvertices ? j : (j - p2.nvertices);
                if (xflags != nullptr)
                    xflags[result.nvertices] = jmod << 1;
                result.vertices[result.nvertices++] = p2.vertices[jmod];
            }
        }
        edge_flag = !edge_flag;
    }

    return result;
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t area(const AABox2<scalar_t> &a)
{
    return (a.max_x - a.min_x) * (a.max_y - a.min_y);
}

// Calculate inner area of a polygon
template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
scalar_t area(const Poly2<scalar_t, MaxPoints> &p)
{
    if (p.nvertices <= 2)
        return 0;

    scalar_t sum = p.vertices[p.nvertices-1].x*p.vertices[0].y // deal with head and tail first
                 - p.vertices[p.nvertices-1].y*p.vertices[0].x;
    for (uint8_t i = 1; i < p.nvertices; i++)
        sum += p.vertices[i-1].x*p.vertices[i].y - p.vertices[i].x*p.vertices[i-1].y;
    return sum / 2;
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t dimension(const AABox2<scalar_t> &a)
{
    scalar_t dx = a.max_x - a.min_x, dy = a.max_y - a.min_y;
    return sqrt(dx * dx + dy * dy);
}

// use Rotating Caliper to find the dimension
template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
scalar_t dimension(const Poly2<scalar_t, MaxPoints> &p, uint8_t &flag1, uint8_t &flag2)
{
    if (p.nvertices <= 1) return 0;
    if (p.nvertices == 2) return distance (p.vertices[0], p.vertices[1]);

    uint8_t u = 0, unext = 1, v = 1, vnext = 2;
    scalar_t dmax = 0, d;
    for (u = 0; u < p.nvertices; u++)
    {
        // find farthest point from current edge
        unext = _mod_inc(u, p.nvertices);
        while (_cross(p.vertices[u], p.vertices[unext], p.vertices[v]) <=
               _cross(p.vertices[u], p.vertices[unext], p.vertices[vnext])
        ) {
            v = vnext;
            vnext = _mod_inc(v, p.nvertices);
        }

        // update max value
        d = distance(p.vertices[u], p.vertices[v]);
        if (d > dmax)
        {
            dmax = d;
            flag1 = u;
            flag2 = v;
        }
        
        d = distance(p.vertices[unext], p.vertices[v]);
        if (d > dmax)
        {
            dmax = d;
            flag1 = unext;
            flag2 = v;
        }
    }
    return dmax;
}

template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
scalar_t dimension(const Poly2<scalar_t, MaxPoints> &p)
{ uint8_t _; return dimension(p, _, _); }

// use Rotating Caliper to find the convex hull of two polygons
// xflags here store the vertices flag
// left 7 bits represent the index of vertex in original polygon and
// right 1 bit indicate whether the vertex is from p1(=1) or p2(=0)
template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, MaxPoints1 + MaxPoints2> merge(
    const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    uint8_t xflags[MaxPoints1 + MaxPoints2] = nullptr
) {
    // find the vertices with max y value, starting from line pointing to -x (angle is -pi)
    uint8_t pidx1, pidx2;
    scalar_t y_max1, y_max2;
    _find_extreme(p1, pidx1, y_max1);
    _find_extreme(p2, pidx2, y_max2);

    // declare variables and functions
    Poly2<scalar_t, MaxPoints1 + MaxPoints2> result;
    scalar_t edge_angle = -_pi; // scan from -pi to pi
    bool _; // temp var
    bool edge_flag = y_max1 > y_max2; // true: current edge on p1 will be present in merged polygon,
                                      //false: current edge on p2 will be present in merged polygon

    const auto register_p1point = [&](uint8_t i1)
    {
        if (xflags != nullptr)
            xflags[result.nvertices] = (i1 << 1) | 1;
        result.vertices[result.nvertices++] = p1.vertices[i1];
    };
    const auto register_p2point = [&](uint8_t i2)
    {
        if (xflags != nullptr)
            xflags[result.nvertices] = i2 << 1;
        result.vertices[result.nvertices++] = p2.vertices[i2];
    };
    const auto register_point = [&](uint8_t i1, uint8_t i2)
    {
        if (edge_flag) register_p1point(i1);
        else register_p2point(i2);
    };

    // check if starting point already build a bridge and if so, find correct edge to start
    if (_check_valid_bridge(p1, p2, pidx1, pidx2, _))
    {
        if (edge_flag)
        {
            uint8_t pidx1_next = _mod_inc(pidx1, p1.nvertices);
            scalar_t angle_br = _slope_pp(p1.vertices[pidx1], p2.vertices[pidx2]);
            scalar_t angle1 = _slope_pp(p1.vertices[pidx1], p1.vertices[pidx1_next]);
            if (angle_br < angle1)
                edge_flag = false;
        }
        else
        {
            uint8_t pidx2_next = _mod_inc(pidx2, p2.nvertices);
            scalar_t angle_br = _slope_pp(p2.vertices[pidx2], p1.vertices[pidx1]);
            scalar_t angle2 =  _slope_pp(p2.vertices[pidx2], p2.vertices[pidx2_next]);
            if (angle_br < angle2)
                edge_flag = true;
        }
    }
    else edge_flag = y_max1 > y_max2;

    // start rotating
    while (true)
    {
        uint8_t pidx1_next = _mod_inc(pidx1, p1.nvertices);
        uint8_t pidx2_next = _mod_inc(pidx2, p2.nvertices);
        scalar_t angle1 = _slope_pp(p1.vertices[pidx1], p1.vertices[pidx1_next]);
        scalar_t angle2 = _slope_pp(p2.vertices[pidx2], p2.vertices[pidx2_next]);

        // compare angles and proceed
        if (edge_angle < angle1 && (angle1 < angle2 || angle2 < edge_angle))
        { // choose p1 edge as part of co-podal pair

            if (edge_flag) register_p1point(pidx1); // Add vertex on current edge

            if (_check_valid_bridge(p1, p2, pidx1_next, pidx2, _))
            {
                // valid bridge, add bridge point
                register_point(pidx1_next, pidx2);
                edge_flag = !edge_flag;
            }

            // update pointer
            pidx1 = _mod_inc(pidx1, p1.nvertices);
            edge_angle = angle1;
        }
        else if (edge_angle < angle2 && (angle2 < angle1 || angle1 < edge_angle))
        { // choose p2 edge as part of co-podal pair

            if (!edge_flag) register_p2point(pidx2); // Add vertex on current edge

            if (_check_valid_bridge(p1, p2, pidx1, pidx2_next, _))
            {
                // valid bridge, add bridge point
                register_point(pidx1, pidx2_next);
                edge_flag = !edge_flag;
            }

            // update pointer
            pidx2 = _mod_inc(pidx2, p2.nvertices);
            edge_angle = angle2;
        }
        else break; // when both angles are not increasing, the loop is finished
    }

    return result;
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
AABox2<scalar_t> merge(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2)
{
    return {
        .min_x = _min(a1.min_x, a2.min_x),
        .max_x = _max(a1.max_x, a2.max_x),
        .min_y = _min(a1.min_y, a2.min_y),
        .max_y = _max(a1.max_y, a2.max_y)
    };
}

// use rotating caliper to find max distance between polygons
// this function use cross products to compare distances which could be faster
template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
scalar_t max_distance(
    const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    uint8_t &flag1, uint8_t &flag2
) {

    uint8_t pidx1, pidx2;
    scalar_t y_max1, y_max2;
    _find_extreme<scalar_t, MaxPoints1, true>(p1, pidx1, y_max1);
    _find_extreme<scalar_t, MaxPoints2, false>(p2, pidx2, y_max2);

    // start rotating to find all intersection points
    scalar_t edge_angle = -_pi; // scan p1 from -pi to pi
    scalar_t dmax = distance(p1.vertices[pidx1], p2.vertices[pidx2]);
    flag1 = pidx1; flag2 = pidx2;
    while (true)
    {
        uint8_t pidx1_next = _mod_inc(pidx1, p1.nvertices);
        uint8_t pidx2_next = _mod_inc(pidx2, p2.nvertices);
        scalar_t angle1 = _slope_pp(p1.vertices[pidx1], p1.vertices[pidx1_next]);
        scalar_t angle2 = _slope_pp(p2.vertices[pidx2_next], p2.vertices[pidx2]);

         // compare angles and proceed
        if (edge_angle < angle1 && (angle1 < angle2 || angle2 < edge_angle))
        { // choose p1 edge as part of anti-podal pair

            scalar_t d = distance(p1.vertices[pidx1_next], p2.vertices[pidx2]);
            if (d > dmax)
            {
                flag1 = pidx1_next;
                dmax = d;
            }

            // update pointer
            pidx1 = _mod_inc(pidx1, p1.nvertices);
            edge_angle = angle1;
        }
        else if (edge_angle < angle2 && (angle2 < angle1 || angle1 < edge_angle))
        { // choose p2 edge as part of anti-podal pair

            scalar_t d = distance(p1.vertices[pidx1], p2.vertices[pidx2_next]);
            if (d > dmax)
            {
                flag2 = pidx2_next;
                dmax = d;
            }

            // update pointer
            pidx2 = _mod_inc(pidx2, p2.nvertices);
            edge_angle = angle2;
        }
        else break; // when both angles are not increasing, the loop is finished
    }

    return dmax;
}

template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
scalar_t max_distance(
    const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2
) { uint8_t _; return max_distance(p1, p2, _, _); }

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t max_distance(const AABox2<scalar_t> &b1, const AABox2<scalar_t> &b2)
{
    if (b1.contains(b2) || b2.contains(b1))
    {
        scalar_t dminmin = distance(
            Point2<scalar_t>{.x = b1.min_x, .y = b1.min_y},
            Point2<scalar_t>{.x = b2.min_x, .y = b2.min_y}
        ), dminmax = distance(
            Point2<scalar_t>{.x = b1.min_x, .y = b1.max_y},
            Point2<scalar_t>{.x = b2.min_x, .y = b2.max_y}
        ), dmaxmin = distance(
            Point2<scalar_t>{.x = b1.max_x, .y = b1.min_y},
            Point2<scalar_t>{.x = b2.max_x, .y = b2.min_y}
        ), dmaxmax = distance(
            Point2<scalar_t>{.x = b1.max_x, .y = b1.max_y},
            Point2<scalar_t>{.x = b2.max_x, .y = b2.max_y}
        );
        return _max(_max(dminmin, dmaxmax), _max(dminmax, dmaxmin));
    }
    else
    {
        AABox2<scalar_t> m = merge(b1, b2);
        return dimension(m);
    }
}

///////////// Custom functions /////////////


template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t iou(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2)
{
    scalar_t area_i = area(intersect(a1, a2));
    scalar_t area_u = area(a1) + area(a2) - area_i;
    return area_i / area_u;
}

// calculating iou of two polygons. xflags is used in intersection caculation
template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
scalar_t iou(const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    uint8_t &nx, uint8_t xflags[MaxPoints1 + MaxPoints2] = nullptr)
{
    auto pi = intersect(p1, p2, xflags);
    nx = pi.nvertices;
    scalar_t area_i = area(pi);
    scalar_t area_u = area(p1) + area(p2) - area_i;
    return area_i / area_u;
}

template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
scalar_t iou(const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2)
{
    uint8_t _; return iou(p1, p2, _, nullptr);
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t giou(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2)
{
    scalar_t area_i = area(intersect(a1, a2));
    scalar_t area_m = area(merge(a1, a2));
    scalar_t area_u = area(a1) + area(a2) - area_i;
    return area_i / area_u - (area_m - area_u) / area_m;
}

template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
scalar_t giou(const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    uint8_t &nx, uint8_t &nm, uint8_t xflags[MaxPoints1 + MaxPoints2] = nullptr, uint8_t mflags[MaxPoints1 + MaxPoints2] = nullptr)
{
    auto pi = intersect(p1, p2, xflags);
    nx = pi.nvertices;
    auto pm = merge(p1, p2, mflags);
    nm = pm.nvertices;

    scalar_t area_i = area(pi);
    scalar_t area_m = area(pm);
    scalar_t area_u = area(p1) + area(p2) - area_i;
    return area_i / area_u - (area_m - area_u) / area_m;
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t diou(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2)
{
    scalar_t iou = iou(a1, a2);
    scalar_t maxd = distance(merge(a1, a2));

    Point2<scalar_t> c1 {.x = (a1.max_x + a1.min_x)/2, .y = (a1.max_y + a1.min_y)/2};
    Point2<scalar_t> c2 {.x = (a2.max_x + a2.min_x)/2, .y = (a2.max_y + a2.min_y)/2};
    scalar_t cd = distance(c1, c2);
    
    return 1 - iou - (cd*cd) / (maxd*maxd);
}

} // namespace d3d

#endif // D3D_GEOMETRY_HPP
