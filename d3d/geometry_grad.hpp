/*
 * Copyright (c) 2019- Yuanxin Zhong. All rights reserved.
 * This file contains implementations of gradient calculation of simple geometries suitable for using in GPU.
 * The gradients are stored in the same data structure as data passed forward.
 */

#ifndef D3D_GEOMETRY_GRAD_HPP
#define D3D_GEOMETRY_GRAD_HPP

#include <d3d/geometry.hpp>

namespace d3d
{

template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
void _max_grad(const T &a, const T &b, const T &g, T &ga, T &gb)
{
    if (a > b) { ga = g; gb = 0; }
    else { ga = 0; gb = g; }
}
template <typename T> constexpr CUDA_CALLABLE_MEMBER inline
void _min_grad(const T &a, const T &b, const T &g, T &ga, T &gb)
{
    if (a < b) { ga = g; gb = 0; }
    else { ga = 0; gb = g; } 
}

///////////// gradient implementation of constructors //////////////

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void line2_from_xyxy_grad(const scalar_t &x1, const scalar_t &y1, const scalar_t &x2, const scalar_t &y2,
    const Line2<scalar_t> &grad, scalar_t &x1_grad, scalar_t &y1_grad, scalar_t &x2_grad, scalar_t &y2_grad)
{
    x1_grad +=  grad.b-y2*grad.c;
    x2_grad += -grad.b+y1*grad.c;
    y1_grad += -grad.a+x2*grad.c;
    y2_grad +=  grad.a-x1*grad.c;
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void line2_from_pp_grad(const Point2<scalar_t> &p1, const Point2<scalar_t> &p2, const Line2<scalar_t> &grad,
    Point2<scalar_t> &grad_p1, Point2<scalar_t> &grad_p2)
{
    return line2_from_xyxy_grad(p1.x, p1.y, p2.x, p2.y, grad, grad_p1.x, grad_p1.y, grad_p2.x, grad_p2.y);
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void poly2_from_aabox2_grad(const AABox2<scalar_t> &a, const Poly2<scalar_t, 4> &grad,
    AABox2<scalar_t> &grad_a)
{
    assert(grad.nvertices == 4);
    grad_a.min_x = grad.vertices[0].x + grad.vertices[3].x;
    grad_a.max_x = grad.vertices[1].x + grad.vertices[2].x;
    grad_a.min_y = grad.vertices[0].y + grad.vertices[1].y;
    grad_a.max_y = grad.vertices[2].y + grad.vertices[3].y;
}

template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
void aabox2_from_poly2_grad(const Poly2<scalar_t, MaxPoints> &p, const AABox2<scalar_t> &grad,
    Poly2<scalar_t, MaxPoints> &grad_p)
{
    scalar_t max_x = p.vertices[0].x, min_x = p.vertices[0].x;
    scalar_t max_y = p.vertices[0].y, min_y = p.vertices[0].y;
    uint8_t imax_x = 0, imin_x = 0, imax_y = 0, imin_y = 0;
    
    // find position for minmax values
    for (uint8_t i = 1; i < p.nvertices; i++)
    {
        if (p.vertices[i].x > max_x) { max_x = p.vertices[i].x; imax_x = i; }
        if (p.vertices[i].x < min_x) { min_x = p.vertices[i].x; imin_x = i; }
        if (p.vertices[i].y > max_y) { max_y = p.vertices[i].y; imax_y = i; }
        if (p.vertices[i].y < min_y) { min_y = p.vertices[i].y; imin_y = i; }
    }

    // initialize grad as zero
    grad_p.nvertices = p.nvertices;
    for (uint8_t i = 0; i < p.nvertices; i++)
        grad_p.vertices[i].x = grad_p.vertices[i].y = 0;

    // pass the gradients
    for (uint8_t i = 0; i < p.nvertices; i++)
    {
        if (i == max_x) grad_p.vertices[i].x += grad.max_x;
        if (i == min_x) grad_p.vertices[i].x += grad.min_x;
        if (i == max_y) grad_p.vertices[i].y += grad.max_y;
        if (i == min_y) grad_p.vertices[i].y += grad.min_y;
    }
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void poly2_from_xywhr_grad(const scalar_t& x, const scalar_t& y,
    const scalar_t& w, const scalar_t& h, const scalar_t& r, const Poly2<scalar_t, 4>& grad,
    scalar_t &grad_x, scalar_t &grad_y, scalar_t &grad_w, scalar_t &grad_h, scalar_t &grad_r)
{
    assert(grad.nvertices == 4);

    grad_x = grad.vertices[0].x + grad.vertices[1].x + grad.vertices[2].x + grad.vertices[3].x;
    grad_y = grad.vertices[0].y + grad.vertices[1].y + grad.vertices[2].y + grad.vertices[3].y;

    scalar_t dxsin_grad = -grad.vertices[0].y + grad.vertices[1].y + grad.vertices[2].y - grad.vertices[3].y;
    scalar_t dxcos_grad = -grad.vertices[0].x + grad.vertices[1].x + grad.vertices[2].x - grad.vertices[3].x;
    scalar_t dysin_grad =  grad.vertices[0].x + grad.vertices[1].x - grad.vertices[2].x - grad.vertices[3].x;
    scalar_t dycos_grad = -grad.vertices[0].y - grad.vertices[1].y + grad.vertices[2].y + grad.vertices[3].y;

    grad_w = (dxsin_grad*sin(r) + dxcos_grad*cos(r))/2;
    grad_h = (dysin_grad*sin(r) + dycos_grad*cos(r))/2;
    grad_r = (dxsin_grad*w*cos(r) - dxcos_grad*w*sin(r) + dysin_grad*h*cos(r) - dysin_grad*h*sin(r))/2;
}

///////////// gradient implementation of functions //////////////

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void intersect_grad(const Line2<scalar_t> &l1, const Line2<scalar_t> &l2, const Point2<scalar_t> &grad,
    Line2<scalar_t> &grad_l1, Line2<scalar_t> &grad_l2)
{
    scalar_t wab = l1.a*l2.b - l2.a*l1.b;
    scalar_t wbc = l1.b*l2.c - l2.b*l1.c;
    scalar_t wca = l1.c*l2.a - l2.c*l1.a;

    scalar_t g_xab = grad.x/wab;
    scalar_t g_yab = grad.y/wab;
    scalar_t g_xyab2 = (wbc*g_xab + wca*g_yab)/wab;

    grad_l1.a = -l2.c*g_yab - l2.b*g_xyab2;
    grad_l1.b =  l2.c*g_xab + l2.a*g_xyab2;
    grad_l1.c =  l2.a*g_yab - l2.b*g_xab;
    grad_l2.a =  l1.c*g_yab + l1.b*g_xyab2;
    grad_l2.b = -l1.c*g_xab - l1.a*g_xyab2;
    grad_l2.c =  l1.b*g_xab - l1.a*g_yab;
}

template <typename scalar_t, uint8_t MaxPoints1, uint8_t MaxPoints2> CUDA_CALLABLE_MEMBER inline
void intersect_grad(const Poly2<scalar_t, MaxPoints1> &p1, const Poly2<scalar_t, MaxPoints2> &p2,
    const Poly2<scalar_t, MaxPoints1 + MaxPoints2> grad, const uint8_t xflags[MaxPoints1 + MaxPoints2],
    Poly2<scalar_t, MaxPoints1> &grad_p1, Poly2<scalar_t, MaxPoints2> &grad_p2
) {
    // initialize output // TODO: check default initialization and remove this part, and then make all grad calculation additive, not assignment
    grad_p1.nvertices = p1.nvertices;
    for (uint8_t i = 0; i < p1.nvertices; i++)
        grad_p1.vertices[i].x = grad_p1.vertices[i].y = 0;

    grad_p2.nvertices = p2.nvertices;
    for (uint8_t i = 0; i < p2.nvertices; i++)
        grad_p2.vertices[i].x = grad_p2.vertices[i].y = 0;

    for (uint8_t i = 0; i < grad.nvertices; i++)
    {
        uint8_t iprev = _mod_dec(i, grad.nvertices);
        if ((xflags[i] & 1) == (xflags[iprev] & 1)) // the intersection vertex is from one of the polygons
        {
            if (xflags[i] & 1) grad_p1[xflags[i] >> 1] = grad[i];
            else grad_p2[xflags[i] >> 1] = grad[i];
        }
        else // the intersection vertex is defined by both polygons
        {
            Line2<scalar_t> edge_next, edge_prev, edge_next_grad, edge_grad_prev;
            if (xflags[i] & 1) // next edge is from p1 and previous edge is from p2
            {
                uint8_t epni = xflags[i] >> 1, epnj = _mod_inc(epni, p1.nvertices);
                uint8_t eppj = xflags[iprev] >> 1, eppi = _mod_dec(eppj, p2.nvertices);
                edge_next = line2_from_pp(p1.vertices[epni], p1.vertices[epnj]);
                edge_prev = line2_from_pp(p2.vertices[eppi], p2.vertices[eppj]);
                intersect_grad(edge_next, edge_prev, grad.vertices[i], edge_next_grad, edge_grad_prev);
                line2_from_pp_grad(p1.vertices[epni], p1.vertices[epnj], edge_next_grad, grad_p1.vertices[epni], grad_p1.vertices[epnj]);
                line2_from_pp_grad(p2.vertices[eppi], p2.vertices[eppj], edge_grad_prev, grad_p2.vertices[eppi], grad_p2.vertices[eppj]);
            }
            else // next edge is from p2 and previous edge is from p1
            {
                uint8_t epni = xflags[i] >> 1, epnj = _mod_inc(epni, p2.nvertices);
                uint8_t eppj = xflags[iprev] >> 1, eppi = _mod_dec(eppj, p1.nvertices);
                edge_next = line2_from_pp(p2.vertices[epni], p2.vertices[epnj]);
                edge_prev = line2_from_pp(p1.vertices[eppi], p1.vertices[eppj]);
                intersect_grad(edge_next, edge_prev, grad.vertices[i], edge_next_grad, edge_grad_prev);
                line2_from_pp_grad(p2.vertices[epni], p2.vertices[epnj], edge_next_grad, grad_p2.vertices[epni], grad_p2.vertices[epnj]);
                line2_from_pp_grad(p1.vertices[eppi], p1.vertices[eppj], edge_grad_prev, grad_p1.vertices[eppi], grad_p1.vertices[eppj]);
            }
        }
    }
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void intersect_grad(const AABox2<scalar_t> &a1, const AABox2<scalar_t> &a2, const AABox2<scalar_t> &grad,
    AABox2<scalar_t> &grad_a1, AABox2<scalar_t> &grad_a2)
{
    _max_grad(a1.min_x, a2.min_x, grad.min_x, grad_a1.min_x, grad_a2.min_x);
    _min_grad(a1.max_x, a2.max_x, grad.max_x, grad_a1.max_x, grad_a2.max_x);
    _max_grad(a1.min_y, a2.min_y, grad.min_y, grad_a1.min_y, grad_a2.min_y);
    _min_grad(a1.max_y, a2.max_y, grad.max_y, grad_a1.max_y, grad_a2.max_y);
}

template <typename scalar_t> CUDA_CALLABLE_MEMBER inline
void area_grad(const AABox2<scalar_t> &a, const scalar_t grad, AABox2<scalar_t> &grad_a)
{
    scalar_t lx = a.max_x - a.min_x;
    scalar_t ly = a.max_y - a.min_y;
    grad_a.max_x =  grad * ly;
    grad_a.min_x = -grad * ly;
    grad_a.max_y =  grad * lx;
    grad_a.min_y = -grad * lx;
}

template <typename scalar_t, uint8_t MaxPoints> CUDA_CALLABLE_MEMBER inline
void area_grad(const Poly2<scalar_t, MaxPoints> &p, const scalar_t grad, Poly2<scalar_t, MaxPoints> &grad_p)
{
    grad_p.nvertices = p.nvertices;

    // deal with head and tail vertex
    uint8_t ilast = p.nvertices-1;
    grad_p.vertices[0].x = grad * (p.vertices[1].y - p.vertices[ilast].y);
    grad_p.vertices[0].y = grad * (p.vertices[ilast].x - p.vertices[1].x);
    grad_p.vertices[ilast].x = grad * (p.vertices[0].y - p.vertices[ilast-1].y);
    grad_p.vertices[ilast].y = grad * (p.vertices[ilast-1].x - p.vertices[0].x);

    // process other vertices
    for (uint8_t i = 1; i < ilast; i++)
    {
        grad_p.vertices[i].x = grad * (p.vertices[i-1].y - p.vertices[i+1].y);
        grad_p.vertices[i].y = grad * (p.vertices[i+1].x - p.vertices[i-1].x);
    }
}

} // namespace d3d

#endif // D3D_GEOMETRY_GRAD_HPP
