/*
 * This file contains implementations of algorithms of simple geometries suitable for using in GPU.
 * One can turn to shapely, boost.geometry or CGAL for a complete functionality of geometry operations.
 * 
 * Predicates:
 *      .contains: whether the shape contains another shape
 *      .intersects: whether the shape has intersection with another shaoe
 * 
 * Functions:
 *      .area: calculate the area of the shape
 *      .distance: calculate the (minimum) distance between two shapes
 *      .max_distance: calculate the (maximum) distance between two shapes
 *      .intersect: calculate the intersection of the two shapes
 *      .iou: calcuate the intersection over union (about area)
 *      .merge: calculate the shape with minimum area that contains the two shapes
 */

// TODO: to implement gradients, we need to reconstruct these class member functions to standalone function
// For example, previously we have poly1.intersect(poly2), we should have intersect(poly1, poly2, out poly) and 
// intersect(poly1, poly2, poly_grad, out poly1_grad, out poly2_grad)

#ifndef D3D_GEOMETRY_HPP
#define D3D_GEOMETRY_HPP

#include <cmath>
#include <limits>
#include "d3d/common.h"

////////////////////////// Helpers //////////////////////////
#ifndef GEOMETRY_EPS
    #define GEOMETRY_EPS 1e-6 // numerical eps used in the library, can be customized before import
#endif

template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _max(T a, T b) { return a > b ? a : b; }
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _min(T a, T b) { return a < b ? a : b; }
constexpr double _pi = 3.14159265358979323846;
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _mod_inc(T i, T n) { return i < n - 1 ? (i + 1) : 0; }
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
T _mod_dec(T i, T n) { return i > 0 ? (i - 1) : (n - 1); }

namespace d3d {

//////////////////// forward declarations //////////////////////
template <typename scalar_t = float> struct Point2;
template <typename scalar_t = float> struct Line2;
template <typename scalar_t = float> struct AABox2;
template <typename scalar_t, uint8_t MaxPoints> struct Poly2;
template <typename scalar_t = float> using Box2 = Poly2<float, 4>;

using Point2f = Point2<float>;
using Line2f = Line2<float>;
using AABox2f = AABox2<float>;
template <uint8_t MaxPoints> using Poly2f = Poly2<float, MaxPoints>;
using Box2f = Box2<float>;

///////////////////// implementations /////////////////////
template <typename scalar_t> struct Point2 // Point in 2D surface
{
    scalar_t x, y;
};

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

    CUDA_CALLABLE_MEMBER inline bool intersects(const AABox2<scalar_t>& a) const
    {
        return (max_x > a.min_x && min_x > a.max_x && max_y < a.min_y && min_y > a.max_y);
    }
};

template <typename scalar_t, uint8_t MaxPoints> struct Poly2 // Convex polygon with no holes
{
    Point2<scalar_t> vertices[MaxPoints]; // vertices in counter-clockwise order
    uint8_t nvertices = 0; // actual number of vertices

    template <uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER
    Poly2<scalar_t, MaxPoints>& operator=(Poly2<scalar_t, MaxPointsOther> const &other)
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
        auto seg = line2_from_pp(vertices[nvertices-1], vertices[0]);
        if (distance(seg, p) > 0) return false;

        // then loop remaining edges
        for (uint8_t i = 1; i < nvertices; i++)
        {
            seg = line2_from_pp(vertices[i-1], vertices[i]);
            if (distance(seg, p) > 0) return false;
        }
        return true;
    }
};

////////////////// constructors ///////////////////

// Note that the order of points matters
template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
Line2<scalar_t> line2_from_xyxy(scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2)
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

// Sutherland-Hodgman https://stackoverflow.com/a/45268241/5960776
// This algorithm is the simplest one, but it's actually O(N*M) complexity
// For more efficient algorithms refer to Rotating Calipers, Sweep Line, etc
template <typename scalar_t, uint8_t MaxPoints, uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, MaxPoints + MaxPointsOther> intersect(
    const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2
) {
    using PolyT = Poly2<scalar_t, MaxPoints + MaxPointsOther>;
    PolyT temp1, temp2; // declare variables to store temporary results
    PolyT *pcut = &(temp1 = p1), *pcur = &temp2; // start with original polygon

    // these flags are saved for backward computation
    // left 16bits for points from p1, right 16bits for points from p2
    // left 15bits from the 16bits are index number, right 1 bit indicate whether point from this polygon is used
    int16_t flag1[MaxPoints + MaxPointsOther], flag2[MaxPoints + MaxPointsOther];
    int16_t *fcut = flag1, *fcur = flag2;

    for (uint8_t j = 0; j < p2.nvertices; j++) // loop over edges of polygon doing cut
    {
        uint8_t jnext = _mod_inc(j, p2.nvertices);
        auto edge = line2_from_pp(p2.vertices[j], p2.vertices[jnext]);

        scalar_t signs[MaxPoints + MaxPointsOther];
        for (uint8_t i = 0; i < pcut->nvertices; i++)
            signs[i] = distance(edge, pcut->vertices[i]);

        for (uint8_t i = 0; i < pcut->nvertices; i++) // loop over edges of polygon to be cut
        {
            if (signs[i] < GEOMETRY_EPS) // eps is used for numerical stable when the boxes are very close
                pcur->vertices[pcur->nvertices++] = pcut->vertices[i];

            uint8_t inext = _mod_inc(i, pcut->nvertices);
            if (signs[i] * signs[inext] < -GEOMETRY_EPS)
            {
                auto cut = line2_from_pp(pcut->vertices[i], pcut->vertices[inext]);
                pcur->vertices[pcur->nvertices++] = intersect(edge, cut);
            }
        }

        PolyT* p = pcut; pcut = pcur; pcur = p; // swap
        pcur->nvertices = 0; // clear current polygon for next iteration
    }
    return *pcut;
}

// Find the intersection point under a bridge over two convex polygons
// p1 is searched from idx1 in clockwise order and p2 is searched from idx2 in counter-clockwise order
// the indices of edges of the intersection are reported by xidx1 and xidx2
// note: index of an edge is the index of its (counter-clockwise) starting point
// If intersection is not found, then false will be returned
template <typename scalar_t, uint8_t MaxPoints, uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER inline // TODO: rename MaxPointsOther to MaxPoints2
bool _find_intersection_under_bridge(const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2,
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
        // std::cout << "down along p2 idx1=" << (int)i1 << ", idx2=" << (int)i2 << std::endl;
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
        // std::cout << "down along p1 idx1=" << (int)i1 << ", idx2=" << (int)i2 << std::endl;
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

template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
void _find_ymax_vertex(const Poly2<scalar_t, MaxPoints> &p, uint8_t &idx, scalar_t &y_max)
{
    idx = 0;
    y_max = p.vertices[0].y;
    for (uint8_t i = 1; i < p.nvertices; i++)
        if (p.vertices[i].y > y_max)
        {
            y_max = p.vertices[i].y;
            idx = i;
        }
}

// check if the bridge defined between two polygon (from p1 to p2) is valid.
// reverse is set to true if the polygons lay in the right of the bridge
template <typename scalar_t, uint8_t MaxPoints, uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER inline
bool _check_valid_bridge(const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2,
    const uint8_t &idx1, const uint8_t &idx2, bool &reverse)
{
    Line2<scalar_t> bridge = line2_from_pp(p1.vertices[idx1], p2.vertices[idx2]);
    scalar_t d1 = distance(bridge, p1.vertices[_mod_dec(idx1, p1.nvertices)]);
    scalar_t d2 = distance(bridge, p1.vertices[_mod_inc(idx1, p1.nvertices)]);
    scalar_t d3 = distance(bridge, p2.vertices[_mod_dec(idx2, p2.nvertices)]);
    scalar_t d4 = distance(bridge, p2.vertices[_mod_inc(idx2, p2.nvertices)]);

    reverse = d1 > 0;
    return d1*d2 > 0 && d2*d3 > 0 && d3*d4 > 0;
}

// Rotating Caliper implementation of intersecting
template <typename scalar_t, uint8_t MaxPoints, uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, MaxPoints + MaxPointsOther> intersect_rc(
    const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2
) {
    // find the vertices with max y value, starting from line pointing to -x (angle is -pi)
    uint8_t pidx1, pidx2; float _;
    _find_ymax_vertex(p1, pidx1, _);
    _find_ymax_vertex(p2, pidx2, _);

    // start rotating to find all intersection points
    float edge_angle = -_pi; // scan from -pi to pi
    bool edge_flag; // true: intersection connection will start with p1, false: start with p2
    uint8_t nx = 0; // number of intersection points
    uint8_t x1indices[MaxPoints + MaxPointsOther + 1], x2indices[MaxPoints + MaxPointsOther + 1];

    while (true)
    {
        // std::cout << "loop " << (int)pidx1 << " " << (int)pidx2 << std::endl;
        uint8_t pidx1_next = _mod_inc(pidx1, p1.nvertices);
        uint8_t pidx2_next = _mod_inc(pidx2, p2.nvertices);
        // TODO: we may face precision problem near horizontal edge
        scalar_t angle1 = atan2(p1.vertices[pidx1_next].y - p1.vertices[pidx1].y,
                                p1.vertices[pidx1_next].x - p1.vertices[pidx1].x);
        scalar_t angle2 = atan2(p2.vertices[pidx2_next].y - p2.vertices[pidx2].y,
                                p2.vertices[pidx2_next].x - p2.vertices[pidx2].x);

        // compare angles and proceed
        if (edge_angle < angle1 && (angle1 < angle2 || angle2 < edge_angle))
        { // choose p1 edge as part of co-podal pair

            bool reverse;
            if (_check_valid_bridge(p1, p2, pidx1_next, pidx2, reverse)) // valid bridge
            {
                // std::cout << "valid bridge pidx1=" << (int)pidx1_next << ", pidx2=" << (int)pidx2 << std::endl;
                uint8_t xidx1, xidx2;
                bool has_intersection = reverse ?
                    _find_intersection_under_bridge(p1, p2, pidx1_next, pidx2, xidx1, xidx2):
                    _find_intersection_under_bridge(p2, p1, pidx2, pidx1_next, xidx2, xidx1);
                if (!has_intersection)
                    return {}; // return empty polygon
                
                // save intersection
                if (nx == 0) edge_flag = !reverse;
                x1indices[nx] = xidx1;
                x2indices[nx] = xidx2;
                nx++;
            }

            // update pointer
            pidx1 = _mod_inc(pidx1, p1.nvertices);
            edge_angle = angle1;
        }
        else if (edge_angle < angle2 && (angle2 < angle1 || angle1 < edge_angle))
        { // choose p2 edge as part of co-podal pair

            bool reverse;
            if (_check_valid_bridge(p1, p2, pidx1, pidx2_next, reverse)) // valid bridge
            {
                // std::cout << "valid bridge pidx1=" << (int)pidx1 << ", pidx2=" << (int)pidx2_next << std::endl;
                uint8_t xidx1, xidx2;
                bool has_intersection = reverse ?
                    _find_intersection_under_bridge(p1, p2, pidx1, pidx2_next, xidx1, xidx2):
                    _find_intersection_under_bridge(p2, p1, pidx2_next, pidx1, xidx2, xidx1);
                if (!has_intersection)
                    return {}; // return empty polygon
                
                // save intersection
                if (nx == 0) edge_flag = !reverse;
                x1indices[nx] = xidx1;
                x2indices[nx] = xidx2;
                nx++;
                // std::cout << "find intersection idx1=" << (int)xidx1 << ", idx2=" << (int)xidx2 << std::endl;
            }

            // update pointer
            pidx2 = _mod_inc(pidx2, p2.nvertices);
            edge_angle = angle2;
        }
        else break; // when both angles are not increasing, the loop is finished
    }

    // add sentinels
    x1indices[nx] = x1indices[0];
    x2indices[nx] = x2indices[0];
    // for (uint8_t i = 0; i < nx; i++)
    //     std::cout << "xidx 1: " << int(x1indices[i]) << ", 2:" << int(x2indices[i]) << std::endl;

    // loop over the intersections to construct the result polygon
    Poly2<scalar_t, MaxPoints + MaxPointsOther> result;
    for (uint8_t i = 0; i < nx; i++)
    {
        // add intersection point
        Line2<scalar_t> l1 = line2_from_pp(p1.vertices[x1indices[i]], p1.vertices[_mod_inc(x1indices[i], p1.nvertices)]);
        Line2<scalar_t> l2 = line2_from_pp(p2.vertices[x2indices[i]], p2.vertices[_mod_inc(x2indices[i], p2.nvertices)]);
        result.vertices[result.nvertices++] = intersect(l1, l2);

        // add points between intersections
        if (edge_flag)
        {
            uint16_t idx_next = x1indices[i+1] >= x1indices[i] ? x1indices[i+1] : (x1indices[i+1] + p1.nvertices);
            for (uint16_t j = x1indices[i] + 1; j <= idx_next; j++)
                result.vertices[result.nvertices++] = p1.vertices[j < p1.nvertices ? j : (j - p1.nvertices)];
        }
        else
        {
            uint16_t idx_next = x2indices[i+1] >= x2indices[i] ? x2indices[i+1] : (x2indices[i+1] + p2.nvertices);
            for (uint16_t j = x2indices[i] + 1; j <= idx_next; j++)
                result.vertices[result.nvertices++] = p2.vertices[j < p2.nvertices ? j : (j - p2.nvertices)];
        }
        // std::cout << (int)result.nvertices << std::endl;
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
scalar_t iou(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2)
{
    scalar_t area_i = area(intersect(a1, a2));
    scalar_t area_u = area(a1) + area(a2) - area_i;
    return area_i / area_u;
}

template <typename scalar_t, uint8_t MaxPoints, uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER inline
scalar_t iou(const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2)
{
    scalar_t area_i = area(intersect(p1, p2));
    scalar_t area_u = area(p1) + area(p2) - area_i;
    return area_i / area_u;
}

// use Rotating Caliper to find the convex hull of two polygons
template <typename scalar_t, uint8_t MaxPoints, uint8_t MaxPointsOther> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, MaxPoints + MaxPointsOther> merge(const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2)
{
    // find the vertices with max y value, starting from line pointing to -x (angle is -pi)
    uint8_t pidx1, pidx2;
    float y_max1, y_max2;
    _find_ymax_vertex(p1, pidx1, y_max1);
    _find_ymax_vertex(p2, pidx2, y_max2);

    // start rotating
    Poly2<scalar_t, MaxPoints + MaxPointsOther> result;
    float edge_angle = -_pi; // scan from -pi to pi
    bool edge_flag = y_max1 > y_max2; // true: current edge on p1 will be present in merged polygon, false: current edge on p2 will be present in merged polygon

    while (true)
    {
        uint8_t pidx1_next = _mod_inc(pidx1, p1.nvertices);
        uint8_t pidx2_next = _mod_inc(pidx2, p2.nvertices);
        scalar_t angle1 = atan2(p1.vertices[pidx1_next].y - p1.vertices[pidx1].y,
                                p1.vertices[pidx1_next].x - p1.vertices[pidx1].x);
        scalar_t angle2 = atan2(p2.vertices[pidx2_next].y - p2.vertices[pidx2].y,
                                p2.vertices[pidx2_next].x - p2.vertices[pidx2].x);

        // compare angles and proceed
        if (edge_angle < angle1 && (angle1 < angle2 || angle2 < edge_angle))
        { // choose p1 edge as part of co-podal pair
            if (edge_flag) // Add vertex on current edge
                result.vertices[result.nvertices++] = p1.vertices[pidx1];

            bool _;
            if (_check_valid_bridge(p1, p2, pidx1_next, pidx2, _))
            {
                // valid bridge, add bridge point
                if (edge_flag)
                    result.vertices[result.nvertices++] = p1.vertices[pidx1_next];
                else
                    result.vertices[result.nvertices++] = p2.vertices[pidx2];
                edge_flag = !edge_flag;
            }

            // update pointer
            pidx1 = _mod_inc(pidx1, p1.nvertices);
            edge_angle = angle1;
        }
        else if (edge_angle < angle2 && (angle2 < angle1 || angle1 < edge_angle))
        { // choose p2 edge as part of co-podal pair
            if (!edge_flag) // Add vertex on current edge
                result.vertices[result.nvertices++] = p2.vertices[pidx2];

            bool _;
            if (_check_valid_bridge(p1, p2, pidx1, pidx2_next, _))
            {
                // valid bridge, add bridge point
                if (edge_flag)
                    result.vertices[result.nvertices++] = p1.vertices[pidx1];
                else
                    result.vertices[result.nvertices++] = p2.vertices[pidx2_next];
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

} // namespace d3d

#endif // D3D_GEOMETRY_HPP
