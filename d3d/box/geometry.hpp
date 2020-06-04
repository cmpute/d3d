/*
 * This file contains implementations of algorithms of simple geometries on stack instead of heap, which are suitable for using in GPU.
 * One can turn to boost.geometry or CGAL for a complete functionality of geometry operations.
 * 
 * TODO: make precision templated
 */

#ifndef D3D_GEOMETRY_HPP
#define D3D_GEOMETRY_HPP

#include <cmath>
#include <limits>
#include "d3d/common.h"

// forward declarations
struct Point2;
struct Line2;
struct AABox2;
template <int MaxPoints> struct Poly2;
struct Box2;


// implementations
struct Point2
{
    float x = 0, y = 0;
    CUDA_CALLABLE_MEMBER Point2() {}
    CUDA_CALLABLE_MEMBER Point2(const float& x_, const float& y_) : x(x_), y(y_) {}
    CUDA_CALLABLE_MEMBER inline Point2 operator+(const Point2& v2) { return Point2(x + v2.x, y + v2.y); }
    CUDA_CALLABLE_MEMBER inline Point2 operator-(const Point2& v2) { return Point2(x - v2.x, y - v2.y); }
};


struct Line2
{
    float a = 0, b = 0, c = 0; // a*x + b*y + c = 0
    CUDA_CALLABLE_MEMBER Line2() {}
    CUDA_CALLABLE_MEMBER Line2(float x1, float y1, float x2, float y2) :
        a(y2-y1), b(x1-x2), c(x2*y1 - x1*y2) {}
    CUDA_CALLABLE_MEMBER Line2(Point2 p1, Point2 p2) : Line2(p1.x, p1.y, p2.x, p2.y) {}

    // Calculate signed distance from point p to a the line
    CUDA_CALLABLE_MEMBER inline float dist(const Point2& p) const { return (a*p.x + b*p.y + c) / sqrt(a*a + b*b); }

    // Calculate intersection point of two lines
    CUDA_CALLABLE_MEMBER inline Point2 intersect(const Line2& l) const
    {
        float w = a*l.b - l.a*b;
        return Point2((b*l.c - l.b*c) / w, (c*l.a - l.c*a) / w); 
    }
};


struct AABox2 // Axis-aligned 2D box for quick calculation
{
    float min_x = 0, max_x = 0;
    float min_y = 0, max_y = 0;

    CUDA_CALLABLE_MEMBER AABox2() {};
    CUDA_CALLABLE_MEMBER inline bool contains(const Point2& p) const;
    CUDA_CALLABLE_MEMBER inline AABox2 intersect(const AABox2& other) const;
    CUDA_CALLABLE_MEMBER inline float area() const;
    CUDA_CALLABLE_MEMBER inline Box2 box() const;
    CUDA_CALLABLE_MEMBER inline float iou(const AABox2 &other) const;
};

template <int MaxPoints> struct Poly2 // Convex polygon with no holes
{
    Point2 vertices[MaxPoints]; // vertices in counter-clockwise order
    size_t nvertices = 0; // actual number of vertices
    CUDA_CALLABLE_MEMBER Poly2() {}

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    void operator=(const Poly2<MaxPointsOther> &other)
    {
        assert(other.nvertices <= MaxPoints);
        nvertices = other.nvertices;
        for(size_t i = 0; i < other.nvertices; i++)
            vertices[i] = other.vertices[i];
    }

    // Calculate inner area
    CUDA_CALLABLE_MEMBER virtual float area() const
    {
        if (nvertices <= 2)
            return 0;

        float sum = 0;
        for (size_t i = 0; i < nvertices; i++)
        {
            int j = (i+1) % nvertices;
            sum += (vertices[i].x*vertices[j].y - vertices[j].x*vertices[i].y);
        }
        return sum / 2;
    }

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    Poly2<MaxPoints + MaxPointsOther> intersect(const Poly2<MaxPointsOther> &other) const
    // XXX: if to make the intersection differentialble, need to store which points are selected
    //      e.g. self points are indexed from 1 to n, other points are indexed from -1 to -n, 0 is reserved for tensor padding
    {
        Poly2<MaxPoints + MaxPointsOther> intersection, temp; // start with original polygon
        intersection = *this;
        for (size_t j = 0; j < other.nvertices; j++) // loop over edges of polygon doing cut
        {
            Line2 edge(other.vertices[j], other.vertices[(j+1) % other.nvertices]);

            float signs[MaxPoints + MaxPointsOther];
            for (size_t i = 0; i < intersection.nvertices; i++) // caculate
                signs[i] = edge.dist(intersection.vertices[i]);

            for (size_t i = 0; i < intersection.nvertices; i++) // loop over edges of polygon to be cut
            {
                if (signs[i] <= 0)
                    temp.vertices[temp.nvertices++] = intersection.vertices[i];
                if (signs[i] * signs[(i+1) % intersection.nvertices] < 0)
                {
                    Line2 cut(intersection.vertices[i], intersection.vertices[(i+1) % intersection.nvertices]);
                    temp.vertices[temp.nvertices++] = edge.intersect(cut);
                }
            }
            intersection = temp; // TODO: implement swap protocol to prevent copy
            temp = Poly2<MaxPoints + MaxPointsOther>();
        }
        return intersection;
    }

    template <int MaxPointsOther> CUDA_CALLABLE_MEMBER
    inline float iou(const Poly2<MaxPointsOther> &other) const
    {
        float area_i = intersect(other).area();
        float area_u = area() + other.area() - area_i;
        return area_i / area_u;
    }

    CUDA_CALLABLE_MEMBER inline bool contains(const Point2& p) const
    {
        for (size_t i = 0; i < nvertices; i++)
        {
            Line2 seg(vertices[i], vertices[(i+1) % nvertices]);
            if (seg.dist(p) > 0) return false;
        }
        return true;
    }

    CUDA_CALLABLE_MEMBER inline AABox2 bbox() const
    {
        AABox2 result;
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


struct Box2 : Poly2<4>
{
    float _area = 0;
    CUDA_CALLABLE_MEMBER Box2() {}
    CUDA_CALLABLE_MEMBER Box2(const float& x, const float& y,
        const float& w, const float& h, const float& r)
    {
        _area = w * h;
        float dxsin = w*sin(r)/2, dxcos = w*cos(r)/2;
        float dysin = h*sin(r)/2, dycos = h*cos(r)/2;
        
        nvertices = 4;
        vertices[0] = Point2(x - dxcos + dysin, y - dxsin - dycos);
        vertices[1] = Point2(x + dxcos + dysin, y + dxsin - dycos);
        vertices[2] = Point2(x + dxcos - dysin, y + dxsin + dycos);
        vertices[3] = Point2(x - dxcos - dysin, y - dxsin + dycos);
    }

    CUDA_CALLABLE_MEMBER float area() const { return _area; }
};


// ---------- AABox2 implementations ----------

CUDA_CALLABLE_MEMBER inline bool AABox2::contains(const Point2& p) const
{
    return p.x > min_x && p.x < max_x && p.y > min_y && p.y < max_y;
}

CUDA_CALLABLE_MEMBER inline AABox2 AABox2::intersect(const AABox2& other) const
{
    AABox2 result;
    // return empty aabox if no intersection
    if (max_x < other.min_x || min_x > other.max_x)
        return result;
    if (max_y < other.min_y || min_y > other.max_y)
        return result;

    result.min_x = fmaxf(min_x, other.min_x);
    result.min_y = fmaxf(min_y, other.min_y);
    result.max_x = fminf(max_x, other.max_x);
    result.max_y = fminf(max_y, other.max_y);
    return result;
}

CUDA_CALLABLE_MEMBER inline float AABox2::area() const
{
    return (max_x - min_x) * (max_y - min_y);
}

CUDA_CALLABLE_MEMBER inline Box2 AABox2::box() const
{
    Box2 result;
    result.nvertices = 4;
    result.vertices[0] = Point2(min_x, min_y);
    result.vertices[1] = Point2(max_x, min_y);
    result.vertices[2] = Point2(max_x, max_y);
    result.vertices[3] = Point2(min_x, max_y);
    result._area = area();
    return result;
}

CUDA_CALLABLE_MEMBER inline float AABox2::iou(const AABox2 &other) const
{
    float area_i = intersect(other).area();
    float area_u = area() + other.area() - area_i;
    return area_i / area_u;
}
// ---------- AABox2 implementations end ----------

#endif // D3D_GEOMETRY_HPP
