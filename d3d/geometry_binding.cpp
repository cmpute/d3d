
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <d3d/geometry.hpp>
#include <d3d/geometry_grad.hpp>

namespace py = pybind11;
using namespace std;
using namespace d3d;
using namespace pybind11::literals;
typedef double T;

template <typename scalar_t, uint8_t MaxPoints> inline
Poly2<scalar_t, MaxPoints> poly_from_points(const vector<Point2<scalar_t>> &points)
{
    Poly2<scalar_t, MaxPoints> b; assert(points.size() <= MaxPoints);
    b.nvertices = points.size();
    for (uint8_t i = 0; i < points.size(); i++)
        b.vertices[i] = points[i];
    return b;
}

PYBIND11_MODULE(geometry, m) {
    m.doc() = "Python binding of the builtin geometry library of d3d, mainly for testing";

    py::class_<Point2<T>>(m, "Point2")
        .def(py::init<>())
        .def(py::init<T, T>())
        .def_readwrite("x", &Point2<T>::x)
        .def_readwrite("y", &Point2<T>::y);
    py::class_<Line2<T>>(m, "Line2")
        .def(py::init<>())
        .def(py::init<T, T, T>())
        .def_readwrite("a", &Line2<T>::a)
        .def_readwrite("b", &Line2<T>::b)
        .def_readwrite("c", &Line2<T>::c);
    py::class_<AABox2<T>>(m, "AABox2")
        .def(py::init<>())
        .def(py::init<T, T, T, T>())
        .def_readwrite("min_x", &AABox2<T>::min_x)
        .def_readwrite("max_x", &AABox2<T>::max_x)
        .def_readwrite("min_y", &AABox2<T>::min_y)
        .def_readwrite("max_y", &AABox2<T>::max_y);
    py::class_<Box2<T>>(m, "Box2")
        .def(py::init<>())
        .def(py::init(&poly_from_points<T, 4>))
        .def_readonly("nvertices", &Box2<T>::nvertices)
        .def_property_readonly("vertices", [](const Box2<T> &b) {
            return vector<Point2<T>>(b.vertices, b.vertices + b.nvertices);});
    py::class_<Poly2<T, 8>>(m, "Poly28")
        .def(py::init<>())
        .def(py::init(&poly_from_points<T, 8>))
        .def_readonly("nvertices", &Poly2<T, 8>::nvertices)
        .def_property_readonly("vertices", [](const Poly2<T, 8> &p) {
            return vector<Point2<T>>(p.vertices, p.vertices + p.nvertices);});

    m.def("line2_from_pp", &d3d::line2_from_pp<T>, "Create a line with two points");
    m.def("line2_from_xyxy", &d3d::line2_from_xyxy<T>, "Create a line with coordinate of two points");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<T, 4>, "Create bounding box of a polygon");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<T, 8>, "Create bounding box of a polygon");
    m.def("poly2_from_aabox2", &d3d::poly2_from_aabox2<T>, "Convert axis aligned box to polygon representation");
    m.def("poly2_from_xywhr", &d3d::poly2_from_xywhr<T>, "Creat a box with specified box parameters");

    m.def("area", py::overload_cast<const AABox2<T>&>(&d3d::area<T>), "Get the area of axis aligned box");
    m.def("area", py::overload_cast<const Box2<T>&>(&d3d::area<T, 4>), "Get the area of box");
    m.def("area", py::overload_cast<const Poly2<T, 8>&>(&d3d::area<T, 8>), "Get the area of polygon");
    m.def("dimension", py::overload_cast<const AABox2<T>&>(&d3d::dimension<T>), "Get the dimension of axis aligned box");
    m.def("dimension", py::overload_cast<const Box2<T>&>(&d3d::dimension<T, 4>), "Get the dimension of box");
    m.def("dimension_", [](const Box2<T>& b){
        uint8_t i1, i2; T v = d3d::dimension(b, i1, i2);
        return make_tuple(v, i1, i2);
    }, "Get the dimension of box");
    m.def("dimension", py::overload_cast<const Poly2<T, 8>&>(&d3d::dimension<T, 8>), "Get the dimension of polygon");
    m.def("dimension_", [](const Poly2<T, 8>& b){
        uint8_t i1, i2; T v = d3d::dimension(b, i1, i2);
        return make_tuple(v, i1, i2);
    }, "Get the dimension of polygon");
    m.def("distance", py::overload_cast<const Point2<T>&, const Point2<T>&>(&d3d::distance<T>),
        "Get the distance between two points");
    m.def("distance", py::overload_cast<const Line2<T>&, const Point2<T>&>(&d3d::distance<T>),
        "Get the distance from a point to a line");
    m.def("intersect", py::overload_cast<const Line2<T>&, const Line2<T>&>(&d3d::intersect<T>),
        "Get the intersection point of two lines");
    m.def("intersect", [](const Box2<T>& b1, const Box2<T>& b2){ return d3d::intersect(b1, b2); },
        "Get the intersection polygon of two boxes");
    m.def("intersect_", [](const Box2<T>& b1, const Box2<T>& b2){
            uint8_t xflags[8];
            auto result = d3d::intersect(b1, b2, xflags);
            vector<uint8_t> flags(xflags, xflags + result.nvertices);
            return make_pair(result, flags);
        }, "Get the intersection polygon of two boxes and return flags");
    m.def("intersect", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::intersect<T>),
        "Get the intersection box of two axis aligned boxes");
    m.def("iou", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::iou<T>),
        "Get the intersection over union of two axis aligned boxes");
    m.def("iou", py::overload_cast<const Box2<T>&, const Box2<T>&>(&d3d::iou<T, 4, 4>),
        "Get the intersection over union of two boxes");
    m.def("iou_", [](const Box2<T>& b1, const Box2<T>& b2){
            uint8_t xflags[8]; uint8_t nx;
            auto result = d3d::iou(b1, b2, nx, xflags);
            vector<uint8_t> flags(xflags, xflags + nx);
            return make_pair(result, flags);
        }, "Get the intersection over union of two boxes and return flags");
    m.def("merge", [](const Box2<T>& b1, const Box2<T>& b2){ return d3d::merge(b1, b2); },
        "Get merged convex hull of two polygons");
    m.def("merge", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::merge<T>),
        "Get bounding box of two axis aligned boxes");
    m.def("max_distance", py::overload_cast<const Box2<T>&, const Box2<T>&>(&d3d::max_distance<T, 4, 4>),
        "Get the max distance between two polygons");
    m.def("max_distance", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::max_distance<T>),
        "Get the max distance between two polygons");

    m.def("line2_from_pp_grad", &d3d::line2_from_pp_grad<T>, "Calculate gradient of line2_from_pp");
    m.def("line2_from_xyxy_grad", &d3d::line2_from_pp_grad<T>, "Calculate gradient of line2_from_xyxy");
    m.def("poly2_from_aabox2_grad", &d3d::poly2_from_aabox2_grad<T>, "Calculate gradient of poly2_from_aabox2");
    m.def("poly2_from_xywhr_grad", [](const T& x, const T& y,
        const T& w, const T& h, const T& r, const Box2<T>& grad){
            T gx, gy, gw, gh, gr;
            poly2_from_xywhr_grad(x, y, w, h, r, grad, gx, gy, gw, gh, gr);
            return make_tuple(gx, gy, gw, gh, gr);
        }, "Calculate gradient of poly2_from_xywhr");
    m.def("aabox2_from_poly2_grad", &d3d::aabox2_from_poly2_grad<T, 4>, "Calculate gradient of aabox2_from_poly2_grad");
    m.def("aabox2_from_poly2_grad", &d3d::aabox2_from_poly2_grad<T, 8>, "Calculate gradient of aabox2_from_poly2_grad");
    m.def("intersect_grad", py::overload_cast<const Line2<T>&, const Line2<T>&, const Point2<T>&, Line2<T>&, Line2<T>&>(&d3d::intersect_grad<T>),
        "Calculate gradient of intersect");

    m.def("area_grad", py::overload_cast<const AABox2<T>&, const T&, AABox2<T>&>(&d3d::area_grad<T>), "Calculate gradient of area");
    m.def("area_grad", py::overload_cast<const Box2<T>&, const T&, Box2<T>&>(&d3d::area_grad<T, 4>), "Calculate gradient of area");
    m.def("area_grad", py::overload_cast<const Poly2<T, 8>&, const T&, Poly2<T, 8>&>(&d3d::area_grad<T, 8>), "Calculate gradient of area");
    m.def("intersect_grad", py::overload_cast<const Line2<T>&, const Line2<T>&, const Point2<T>&, Line2<T>&, Line2<T>&>(&d3d::intersect_grad<T>),
        "Calculate gradient of intersect()");
    m.def("intersect_grad", [](const Box2<T>& b1, const Box2<T>& b2,
        const Poly2<T, 8>& grad, array<uint8_t, 8> xflags, Box2<T>& grad_p1, Box2<T>& grad_p2){
            intersect_grad(b1, b2, grad, xflags.data(), grad_p1, grad_p2);
        }, "Calculate gradient of intersect()");
    m.def("intersect_grad", py::overload_cast<const AABox2<T>&, const AABox2<T>&, const AABox2<T>&, AABox2<T>&, AABox2<T>&>(&d3d::intersect_grad<T>),
        "Calculate gradient of intersect()");
}