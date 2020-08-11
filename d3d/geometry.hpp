/*
 * This file contains implementations of algorithms of simple geometries suitable for using in GPU.
 * One can turn to boost.geometry or CGAL for a complete functionality of geometry operations.
 */

// TODO: convert operators into separate functions, but predicates remains inside struct
// predicates are methods producing bool result, including intersects, contains.

// TODO: to implement gradients, we need to reconstruct these class member functions to standalone function
// For example, previously we have poly1.intersect(poly2), we should have intersect(poly1, poly2, out poly) and 
// intersect(poly1, poly2, poly_grad, out poly1_grad, out poly2_grad)

#ifndef D3D_GEOMETRY_HPP
#define D3D_GEOMETRY_HPP

#include <cmath>
#include <limits>
#include "d3d/common.h"

#ifndef GEOMETRY_EPS
    #define GEOMETRY_EPS 1e-6 // numerical eps used in the library
#endif

#define _max(a, b) ( ((a) > (b)) ? (a) : (b) )
#define _min(a, b) ( ((a) < (b)) ? (a) : (b) )

namespace d3d {

//////////////////// forward declarations //////////////////////
template <typename scalar_t = float> struct Point2;
template <typename scalar_t = float> struct Line2;
template <typename scalar_t = float> struct AABox2;
template <typename scalar_t, int MaxPoints> struct Poly2;
template <typename scalar_t = float> using Box2 = Poly2<float, 4>;

using Point2f = Point2<float>;
using Line2f = Line2<float>;
using AABox2f = AABox2<float>;
template <int MaxPoints> using Poly2f = Poly2<float, MaxPoints>;
using Box2f = Box2<float>;

///////////////////// implementations /////////////////////
template <typename scalar_t> struct Point2 // Point in 2D surface
{
    scalar_t x, y;
};

template <typename scalar_t> struct Line2 // Infinite but directional line
{
    scalar_t a, b, c; // a*x + b*y + c = 0
};

template <typename scalar_t> struct AABox2 // Axis-aligned 2D box for quick calculation
{
    scalar_t min_x, max_x, min_y, max_y;

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2<scalar_t>& p) const
    {
        return p.x > min_x && p.x < max_x && p.y > min_y && p.y < max_y;
    }
};

template <typename scalar_t, int MaxPoints> struct Poly2 // Convex polygon with no holes
{
    Point2<scalar_t> vertices[MaxPoints]; // vertices in counter-clockwise order
    size_t nvertices = 0; // actual number of vertices

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    Poly2<scalar_t, MaxPoints>& operator=(Poly2<scalar_t, MaxPointsOther> const &other)
    {
        assert(other.nvertices <= MaxPoints);
        nvertices = other.nvertices;
        for (size_t i = 0; i < other.nvertices; i++)
            vertices[i] = other.vertices[i];
        return *this;
    }

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2<scalar_t>& p) const
    { 
        // deal with head and tail first
        auto seg = line2_from_pp(vertices[nvertices-1], vertices[0]);
        if (distance(seg, p) > 0) return false;

        // then loop remaining edges
        for (size_t i = 1; i < nvertices; i++)
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
template <typename scalar_t, int MaxPoints> CUDA_CALLABLE_MEMBER inline
AABox2<scalar_t> aabox2_from_poly2(const Poly2<scalar_t, MaxPoints> &p)
{
    AABox2<scalar_t> result {
        .min_x = p.vertices[0].x, .max_x = p.vertices[0].x,
        .min_y = p.vertices[0].y, .max_y = p.vertices[0].y
    };

    for (size_t i = 1; i < p.nvertices; i++)
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

//////////////////// operations ///////////////////

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
template <typename scalar_t, int MaxPoints, int MaxPointsOther> CUDA_CALLABLE_MEMBER inline
Poly2<scalar_t, MaxPoints + MaxPointsOther> intersect(const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2)
// XXX: if to make the intersection differentialble, need to store which points are selected
//      e.g. self points are indexed from 1 to n, other points are indexed from -1 to -n, 0 is reserved for tensor padding
//      a solution is using 32bit integer to store the flag, 15bit for each polygon point index and leftmost 2bits for mode:
//           (10 means poly1 vertex, 01 means poly2 vertex, 11 means intersection of two edges)
{
    using PolyT = Poly2<scalar_t, MaxPoints + MaxPointsOther>;
    PolyT temp1, temp2; // declare variables to store temporary results
    PolyT *pcut = &(temp1 = p1), *ptemp = &temp2; // start with original polygon

    for (size_t j = 0; j < p2.nvertices; j++) // loop over edges of polygon doing cut
    {
        size_t jnext = (j == p2.nvertices-1) ? 0 : j+1;
        auto edge = line2_from_pp(p2.vertices[j], p2.vertices[jnext]);

        scalar_t signs[MaxPoints + MaxPointsOther];
        for (size_t i = 0; i < pcut->nvertices; i++)
            signs[i] = distance(edge, pcut->vertices[i]);

        for (size_t i = 0; i < pcut->nvertices; i++) // loop over edges of polygon to be cut
        {
            if (signs[i] < GEOMETRY_EPS) // eps is used for numerical stable when the boxes are very close
                ptemp->vertices[ptemp->nvertices++] = pcut->vertices[i];

            size_t inext = (i == pcut->nvertices-1) ? 0 : i+1;
            if (signs[i] * signs[inext] < -GEOMETRY_EPS)
            {
                auto cut = line2_from_pp(pcut->vertices[i], pcut->vertices[inext]);
                ptemp->vertices[ptemp->nvertices++] = intersect(edge, cut);
            }
        }

        PolyT* p = pcut; pcut = ptemp; ptemp = p; // swap
        ptemp->nvertices = 0;
    }
    return *pcut;
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
scalar_t area(const AABox2<scalar_t> &a)
{
    return (a.max_x - a.min_x) * (a.max_y - a.min_y);
}

// Calculate inner area of a polygon
template <typename scalar_t, int MaxPoints> CUDA_CALLABLE_MEMBER inline
scalar_t area(const Poly2<scalar_t, MaxPoints> &p)
{
    if (p.nvertices <= 2)
        return 0;

    scalar_t sum = p.vertices[p.nvertices-1].x*p.vertices[0].y // deal with head and tail first
                 - p.vertices[p.nvertices-1].y*p.vertices[0].x;
    for (size_t i = 1; i < p.nvertices; i++)
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

template <typename scalar_t, int MaxPoints, int MaxPointsOther> CUDA_CALLABLE_MEMBER inline
scalar_t iou(const Poly2<scalar_t, MaxPoints> &p1, const Poly2<scalar_t, MaxPointsOther> &p2)
{
    scalar_t area_i = area(intersect(p1, p2));
    scalar_t area_u = area(p1) + area(p2) - area_i;
    return area_i / area_u;
}

} // namespace d3d

#endif // D3D_GEOMETRY_HPP
