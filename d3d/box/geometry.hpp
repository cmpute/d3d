/*
 * This file contains implementations of algorithms of simple geometries suitable for using in GPU.
 * One can turn to boost.geometry or CGAL for a complete functionality of geometry operations.
 */

#ifndef D3D_GEOMETRY_HPP
#define D3D_GEOMETRY_HPP

#include <cmath>
#include <limits>
#include "d3d/common.h"

// forward declarations
template <typename scalar_t = float> struct Point2;
template <typename scalar_t = float> struct Line2;
template <typename scalar_t = float> struct AABox2;
template <typename scalar_t, int MaxPoints> struct Poly2;
template <typename scalar_t = float> struct Box2;

using Point2f = Point2<float>;
using Line2f = Line2<float>;
using AABox2f = AABox2<float>;
template <int MaxPoints> using Poly2f = Poly2<float, MaxPoints>;
using Box2f = Box2<float>;

// implementations
template <typename scalar_t> struct Point2
{
    scalar_t x = NAN, y = NAN;
    CUDA_CALLABLE_MEMBER Point2() {}
    CUDA_CALLABLE_MEMBER Point2(const scalar_t& x_, const scalar_t& y_) : x(x_), y(y_) {}
    CUDA_CALLABLE_MEMBER inline Point2<scalar_t> operator+(const Point2<scalar_t>& v2) { return Point2<scalar_t>(x + v2.x, y + v2.y); }
    CUDA_CALLABLE_MEMBER inline Point2<scalar_t> operator-(const Point2<scalar_t>& v2) { return Point2<scalar_t>(x - v2.x, y - v2.y); }
};


template <typename scalar_t> struct Line2
{
    scalar_t a = NAN, b = NAN, c = NAN; // a*x + b*y + c = 0
    CUDA_CALLABLE_MEMBER Line2() {}
    CUDA_CALLABLE_MEMBER Line2(scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2) :
        a(y2-y1), b(x1-x2), c(x2*y1 - x1*y2) {}
    CUDA_CALLABLE_MEMBER Line2(Point2<scalar_t> p1, Point2<scalar_t> p2) : Line2(p1.x, p1.y, p2.x, p2.y) {}

    // Calculate signed distance from point p to the line
    CUDA_CALLABLE_MEMBER inline scalar_t dist(const Point2<scalar_t>& p) const { return (a*p.x + b*p.y + c) / sqrt(a*a + b*b); }

    // Calculate intersection point of two lines
    CUDA_CALLABLE_MEMBER inline Point2<scalar_t> intersect(const Line2<scalar_t>& l) const
    {
        scalar_t w = a*l.b - l.a*b;
        return Point2<scalar_t>((b*l.c - l.b*c) / w, (c*l.a - l.c*a) / w); 
    }
};


template <typename scalar_t> struct AABox2 // Axis-aligned 2D box for quick calculation
{
    scalar_t min_x = NAN, max_x = NAN;
    scalar_t min_y = NAN, max_y = NAN;

    CUDA_CALLABLE_MEMBER AABox2() {};

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2<scalar_t>& p) const
    {
        return p.x > min_x && p.x < max_x && p.y > min_y && p.y < max_y;
    }
    CUDA_CALLABLE_MEMBER inline AABox2<scalar_t> intersect(const AABox2<scalar_t>& other) const
    {
        AABox2<scalar_t> result;
        // return empty aabox if no intersection
        if (max_x < other.min_x || min_x > other.max_x ||
            max_y < other.min_y || min_y > other.max_y)
        {
            result.min_x = result.min_y = 0;
            result.max_x = result.max_y = 0;
        }
        else
        {
            result.min_x = fmaxf(min_x, other.min_x);
            result.min_y = fmaxf(min_y, other.min_y);
            result.max_x = fminf(max_x, other.max_x);
            result.max_y = fminf(max_y, other.max_y);
        }
        return result;
    }
    CUDA_CALLABLE_MEMBER inline scalar_t area() const
    {
        return (max_x - min_x) * (max_y - min_y);
    }
    CUDA_CALLABLE_MEMBER inline scalar_t iou(const AABox2<scalar_t> &other) const
    {
        scalar_t area_i = intersect(other).area();
        scalar_t area_u = area() + other.area() - area_i;
        return area_i / area_u;
    }
    CUDA_CALLABLE_MEMBER inline Box2<scalar_t> box() const;
};

template <typename scalar_t, int MaxPoints> struct Poly2 // Convex polygon with no holes
{
    Point2<scalar_t> vertices[MaxPoints]; // vertices in counter-clockwise order
    size_t nvertices = 0; // actual number of vertices
    CUDA_CALLABLE_MEMBER Poly2() {}

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    Poly2<scalar_t, MaxPoints>& operator=(Poly2<scalar_t, MaxPointsOther> const &other)
    {
        assert(other.nvertices <= MaxPoints);
        nvertices = other.nvertices;
        for (size_t i = 0; i < other.nvertices; i++)
            vertices[i] = other.vertices[i];
        return *this;
    }

    // Calculate inner area
    CUDA_CALLABLE_MEMBER scalar_t area() const
    {
        if (nvertices <= 2)
            return 0;

        scalar_t sum = vertices[nvertices - 1].x*vertices[0].y // deal with head and tail first
                     - vertices[0].x*vertices[nvertices - 1].y;
        for (size_t i = 1; i < nvertices; i++)
            sum += vertices[i-1].x*vertices[i].y - vertices[i].x*vertices[i-1].y;
        return sum / 2;
    }

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    Poly2<scalar_t, MaxPoints + MaxPointsOther> intersect(const Poly2<scalar_t, MaxPointsOther> &other) const
    // XXX: if to make the intersection differentialble, need to store which points are selected
    //      e.g. self points are indexed from 1 to n, other points are indexed from -1 to -n, 0 is reserved for tensor padding
    {
        using PolyT = Poly2<scalar_t, MaxPoints + MaxPointsOther>;
        PolyT temp1, temp2; // declare variables to store temporary results
        PolyT *pcut = &(temp1 = *this), *ptemp = &temp2; // start with original polygon
        for (size_t j = 0; j < other.nvertices; j++) // loop over edges of polygon doing cut
        {
            Line2<scalar_t> edge(other.vertices[j], other.vertices[(j+1) % other.nvertices]);

            scalar_t signs[MaxPoints + MaxPointsOther];
            for (size_t i = 0; i < pcut->nvertices; i++) // caculate
                signs[i] = edge.dist(pcut->vertices[i]);

            for (size_t i = 0; i < pcut->nvertices; i++) // loop over edges of polygon to be cut
            {
                if (signs[i] <= 0)
                    ptemp->vertices[ptemp->nvertices++] = pcut->vertices[i];

                size_t inext = i == (pcut->nvertices-1) ? 0 : i+1;
                if (signs[i] * signs[inext] < 0)
                {
                    Line2<scalar_t> cut(pcut->vertices[i], pcut->vertices[inext]);
                    ptemp->vertices[ptemp->nvertices++] = edge.intersect(cut);
                }
            }

            PolyT* p = pcut; pcut = ptemp; ptemp = p; // swap
            ptemp->nvertices = 0;
        }
        return *pcut;
    }

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    inline scalar_t iou(const Poly2<scalar_t, MaxPointsOther> &other) const
    {
        scalar_t area_i = intersect(other).area();
        scalar_t area_u = area() + other.area() - area_i;
        return area_i / area_u;
    }

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2<scalar_t>& p) const
    { 
        // deal with head and tail first
        if (Line2<scalar_t>(vertices[nvertices-1], vertices[0]).dist(p) > 0)
            return false;
        // then loop remaining edges
        for (size_t i = 1; i < nvertices; i++)
        {
            Line2<scalar_t> seg(vertices[i-1], vertices[i]);
            if (seg.dist(p) > 0) return false;
        }
        return true;
    }

    CUDA_CALLABLE_MEMBER inline AABox2<scalar_t> bbox() const
    {
        AABox2<scalar_t> result;
        result.min_x = result.max_x = vertices[0].x;
        result.min_y = result.max_y = vertices[0].y;

        for (size_t i = 1; i < nvertices; i++)
        {
            result.min_x = fminf(vertices[i].x, result.min_x);
            result.max_x = fmaxf(vertices[i].x, result.max_x);
            result.min_y = fminf(vertices[i].y, result.min_y);
            result.max_y = fmaxf(vertices[i].y, result.max_y);
        }
        return result;
    }
};

template <typename scalar_t> struct Box2 : Poly2<scalar_t, 4>
{
    using Poly2<scalar_t, 4>::nvertices;
    using Poly2<scalar_t, 4>::vertices;

    CUDA_CALLABLE_MEMBER Box2() {}
    CUDA_CALLABLE_MEMBER Box2(const scalar_t& x, const scalar_t& y,
        const scalar_t& w, const scalar_t& h, const scalar_t& r)
    {
        scalar_t dxsin = w*sin(r)/2, dxcos = w*cos(r)/2;
        scalar_t dysin = h*sin(r)/2, dycos = h*cos(r)/2;
        
        nvertices = 4;
        vertices[0] = Point2<scalar_t>(x - dxcos + dysin, y - dxsin - dycos);
        vertices[1] = Point2<scalar_t>(x + dxcos + dysin, y + dxsin - dycos);
        vertices[2] = Point2<scalar_t>(x + dxcos - dysin, y + dxsin + dycos);
        vertices[3] = Point2<scalar_t>(x - dxcos - dysin, y - dxsin + dycos);
    }
};

#endif // D3D_GEOMETRY_HPP
