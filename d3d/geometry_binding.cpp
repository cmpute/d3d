
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
        .def_readwrite("y", &Point2<T>::y)
        .def("__str__", py::overload_cast<const Point2<T>&>(&d3d::to_string<T>))
        .def("__repr__", py::overload_cast<const Point2<T>&>(&d3d::pprint<T>));
    py::class_<Line2<T>>(m, "Line2")
        .def(py::init<>())
        .def(py::init<T, T, T>())
        .def_readwrite("a", &Line2<T>::a)
        .def_readwrite("b", &Line2<T>::b)
        .def_readwrite("c", &Line2<T>::c)
        .def("__str__", py::overload_cast<const Line2<T>&>(&d3d::to_string<T>))
        .def("__repr__", py::overload_cast<const Line2<T>&>(&d3d::pprint<T>));
    py::class_<AABox2<T>>(m, "AABox2")
        .def(py::init<>())
        .def(py::init<T, T, T, T>())
        .def_readwrite("min_x", &AABox2<T>::min_x)
        .def_readwrite("max_x", &AABox2<T>::max_x)
        .def_readwrite("min_y", &AABox2<T>::min_y)
        .def_readwrite("max_y", &AABox2<T>::max_y)
        .def("__str__", py::overload_cast<const AABox2<T>&>(&d3d::to_string<T>))
        .def("__repr__", py::overload_cast<const AABox2<T>&>(&d3d::pprint<T>));
    py::class_<Quad2<T>>(m, "Quad2")
        .def(py::init<>())
        .def(py::init(&poly_from_points<T, 4>))
        .def_readonly("nvertices", &Quad2<T>::nvertices)
        .def_property_readonly("vertices", [](const Quad2<T> &b) {
            return vector<Point2<T>>(b.vertices, b.vertices + b.nvertices);})
        .def("__str__", py::overload_cast<const Quad2<T>&>(&d3d::to_string<T, 4>))
        .def("__repr__", py::overload_cast<const Quad2<T>&>(&d3d::pprint<T, 4>));
    py::class_<Poly2<T, 8>>(m, "Poly28")
        .def(py::init<>())
        .def(py::init(&poly_from_points<T, 8>))
        .def_readonly("nvertices", &Poly2<T, 8>::nvertices)
        .def_property_readonly("vertices", [](const Poly2<T, 8> &p) {
            return vector<Point2<T>>(p.vertices, p.vertices + p.nvertices);})
        .def("__str__", py::overload_cast<const Poly2<T, 8>&>(&d3d::to_string<T, 8>))
        .def("__repr__", py::overload_cast<const Poly2<T, 8>&>(&d3d::pprint<T, 8>));

    // constructors

    m.def("line2_from_pp", &d3d::line2_from_pp<T>, "Create a line with two points");
    m.def("line2_from_xyxy", &d3d::line2_from_xyxy<T>, "Create a line with coordinate of two points");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<T, 4>, "Create bounding box of a polygon");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<T, 8>, "Create bounding box of a polygon");
    m.def("poly2_from_aabox2", &d3d::poly2_from_aabox2<T>, "Convert axis aligned box to polygon representation");
    m.def("poly2_from_xywhr", &d3d::poly2_from_xywhr<T>, "Creat a box with specified box parameters");

    // functions

    m.def("area", py::overload_cast<const AABox2<T>&>(&d3d::area<T>), "Get the area of axis aligned box");
    m.def("area", py::overload_cast<const Quad2<T>&>(&d3d::area<T, 4>), "Get the area of box");
    m.def("area", py::overload_cast<const Poly2<T, 8>&>(&d3d::area<T, 8>), "Get the area of polygon");
    m.def("dimension", py::overload_cast<const AABox2<T>&>(&d3d::dimension<T>), "Get the dimension of axis aligned box");
    m.def("dimension", py::overload_cast<const Quad2<T>&>(&d3d::dimension<T, 4>), "Get the dimension of box");
    m.def("dimension_", [](const Quad2<T>& b){
        uint8_t i1, i2; T v = d3d::dimension(b, i1, i2);
        return make_tuple(v, i1, i2);
    }, "Get the dimension of box");
    m.def("dimension", py::overload_cast<const Poly2<T, 8>&>(&d3d::dimension<T, 8>), "Get the dimension of polygon");
    m.def("dimension_", [](const Poly2<T, 8>& b){
        uint8_t i1, i2; T v = d3d::dimension(b, i1, i2);
        return make_tuple(v, i1, i2);
    }, "Get the dimension of polygon");
    m.def("center", py::overload_cast<const AABox2<T>&>(&d3d::center<T>), "Get the center point of axis aligned box");
    m.def("center", py::overload_cast<const Quad2<T>&>(&d3d::center<T, 4>), "Get the center point of box");
    m.def("center", py::overload_cast<const Poly2<T, 8>&>(&d3d::center<T, 8>), "Get the center point of polygon");
    m.def("centroid", py::overload_cast<const AABox2<T>&>(&d3d::centroid<T>), "Get the centroid point of axis aligned box");
    m.def("centroid", py::overload_cast<const Quad2<T>&>(&d3d::centroid<T, 4>), "Get the centroid point of box");
    m.def("centroid", py::overload_cast<const Poly2<T, 8>&>(&d3d::centroid<T, 8>), "Get the centroid point of polygon");

    // operators

    m.def("distance", py::overload_cast<const Point2<T>&, const Point2<T>&>(&d3d::distance<T>),
        "Get the distance between two points");
    m.def("distance", py::overload_cast<const Line2<T>&, const Point2<T>&>(&d3d::distance<T>),
        "Get the distance from a point to a line");
    m.def("intersect", py::overload_cast<const Line2<T>&, const Line2<T>&>(&d3d::intersect<T>),
        "Get the intersection point of two lines");
    m.def("intersect", [](const Quad2<T>& b1, const Quad2<T>& b2){ return d3d::intersect(b1, b2); },
        "Get the intersection polygon of two boxes");
    m.def("intersect_", [](const Quad2<T>& b1, const Quad2<T>& b2){
            uint8_t xflags[8];
            auto result = d3d::intersect(b1, b2, xflags);
            vector<uint8_t> xflags_v(xflags, xflags + result.nvertices);
            return make_tuple(result, xflags_v);
        }, "Get the intersection polygon of two boxes and return flags");
    m.def("intersect", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::intersect<T>),
        "Get the intersection box of two axis aligned boxes");
    m.def("merge", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::merge<T>),
        "Get bounding box of two axis aligned boxes");
    m.def("merge", [](const Quad2<T>& b1, const Quad2<T>& b2){ return d3d::merge(b1, b2); },
        "Get merged convex hull of two polygons");
    m.def("merge_", [](const Quad2<T>& b1, const Quad2<T>& b2){
            uint8_t mflags[8];
            auto result = d3d::intersect(b1, b2, mflags);
            vector<uint8_t> mflags_v(mflags, mflags + result.nvertices);
            return make_tuple(result, mflags_v);
        }, "Get merged convex hull of two polygons");
    m.def("max_distance", py::overload_cast<const Quad2<T>&, const Quad2<T>&>(&d3d::max_distance<T, 4, 4>),
        "Get the max distance between two polygons");
    m.def("max_distance", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::max_distance<T>),
        "Get the max distance between two polygons");
    m.def("iou", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::iou<T>),
        "Get the intersection over union of two axis aligned boxes");
    m.def("iou", py::overload_cast<const Quad2<T>&, const Quad2<T>&>(&d3d::iou<T, 4, 4>),
        "Get the intersection over union of two boxes");
    m.def("iou_", [](const Quad2<T>& b1, const Quad2<T>& b2){
            uint8_t xflags[8]; uint8_t nx;
            auto result = d3d::iou(b1, b2, nx, xflags);
            vector<uint8_t> xflags_v(xflags, xflags + nx);
            return make_tuple(result, xflags_v);
        }, "Get the intersection over union of two boxes and return flags");
    m.def("giou", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::giou<T>),
        "Get the generalized intersection over union of two axis aligned boxes");
    m.def("giou", py::overload_cast<const Quad2<T>&, const Quad2<T>&>(&d3d::giou<T, 4, 4>),
        "Get the generalized intersection over union of two boxes");
    m.def("giou_", [](const Quad2<T>& b1, const Quad2<T>& b2){
            uint8_t xflags[8], mflags[8]; uint8_t nx, nm;
            auto result = d3d::giou(b1, b2, nx, nm, xflags, mflags);
            vector<uint8_t> xflags_v(xflags, xflags + nx);
            vector<uint8_t> mflags_v(mflags, mflags + nm);
            return make_tuple(result, xflags_v, mflags_v);
        }, "Get the generalized intersection over union of two boxes and return flags");
    m.def("diou", py::overload_cast<const AABox2<T>&, const AABox2<T>&>(&d3d::diou<T>),
        "Get the distance intersection over union of two axis aligned boxes");
    m.def("diou", py::overload_cast<const Quad2<T>&, const Quad2<T>&>(&d3d::diou<T, 4, 4>),
        "Get the distance intersection over union of two boxes");
    m.def("diou_", [](const Quad2<T>& b1, const Quad2<T>& b2){
            uint8_t xflags[8]; uint8_t nx, dflag1, dflag2;
            auto result = d3d::diou(b1, b2, nx, dflag1, dflag2, xflags);
            vector<uint8_t> xflags_v(xflags, xflags + nx);
            return make_tuple(result, xflags_v, dflag1, dflag2);
        }, "Get the distance intersection over union of two boxes and return flags");

    // gradient functions from geometry_grad.hpp
    // gradient of constructors

    m.def("line2_from_pp_grad", &d3d::line2_from_pp_grad<T>, "Calculate gradient of line2_from_pp()");
    m.def("line2_from_xyxy_grad", &d3d::line2_from_pp_grad<T>, "Calculate gradient of line2_from_xyxy()");
    m.def("poly2_from_aabox2_grad", &d3d::poly2_from_aabox2_grad<T>, "Calculate gradient of poly2_from_aabox2()");
    m.def("poly2_from_xywhr_grad", [](const T& x, const T& y,
        const T& w, const T& h, const T& r, const Quad2<T>& grad){
            T gx = 0, gy = 0, gw = 0, gh = 0, gr = 0;
            poly2_from_xywhr_grad(x, y, w, h, r, grad, gx, gy, gw, gh, gr);
            return make_tuple(gx, gy, gw, gh, gr);
        }, "Calculate gradient of poly2_from_xywhr");
    m.def("aabox2_from_poly2_grad", &d3d::aabox2_from_poly2_grad<T, 4>, "Calculate gradient of aabox2_from_poly2_grad()");
    m.def("aabox2_from_poly2_grad", &d3d::aabox2_from_poly2_grad<T, 8>, "Calculate gradient of aabox2_from_poly2_grad()");
    m.def("intersect_grad", py::overload_cast<const Line2<T>&, const Line2<T>&, const Point2<T>&, Line2<T>&, Line2<T>&>(&d3d::intersect_grad<T>),
        "Calculate gradient of intersect");

    // gradient of functions

    m.def("area_grad", py::overload_cast<const AABox2<T>&, const T&, AABox2<T>&>(&d3d::area_grad<T>), "Calculate gradient of area()");
    m.def("area_grad", py::overload_cast<const Quad2<T>&, const T&, Quad2<T>&>(&d3d::area_grad<T, 4>), "Calculate gradient of area()");
    m.def("area_grad", py::overload_cast<const Poly2<T, 8>&, const T&, Poly2<T, 8>&>(&d3d::area_grad<T, 8>), "Calculate gradient of area()");
    m.def("dimension_grad", py::overload_cast<const AABox2<T>&, const T&, AABox2<T>&>(&d3d::dimension_grad<T>), "Calculate gradient of dimension()");
    m.def("dimension_grad", py::overload_cast<const Quad2<T>&, const T&, const uint8_t&, const uint8_t&, Quad2<T>&>(&d3d::dimension_grad<T, 4>),
    "Calculate gradient of dimension()");
    m.def("dimension_grad", py::overload_cast<const Poly2<T, 8>&, const T&, const uint8_t&, const uint8_t&, Poly2<T, 8>&>(&d3d::dimension_grad<T, 8>),
    "Calculate gradient of dimension()");
    m.def("center_grad", py::overload_cast<const AABox2<T>&, const Point2<T>&, AABox2<T>&>(&d3d::center_grad<T>), "Calculate gradient of center()");
    m.def("center_grad", py::overload_cast<const Quad2<T>&, const Point2<T>&, Quad2<T>&>(&d3d::center_grad<T, 4>), "Calculate gradient of center()");
    m.def("center_grad", py::overload_cast<const Poly2<T, 8>&, const Point2<T>&, Poly2<T, 8>&>(&d3d::center_grad<T, 8>), "Calculate gradient of center()");
    m.def("centroid_grad", py::overload_cast<const AABox2<T>&, const Point2<T>&, AABox2<T>&>(&d3d::centroid_grad<T>), "Calculate gradient of centroid()");
    m.def("centroid_grad", py::overload_cast<const Quad2<T>&, const Point2<T>&, Quad2<T>&>(&d3d::centroid_grad<T, 4>), "Calculate gradient of centroid()");
    m.def("centroid_grad", py::overload_cast<const Poly2<T, 8>&, const Point2<T>&, Poly2<T, 8>&>(&d3d::centroid_grad<T, 8>), "Calculate gradient of centroid()");

    // gradient of operators

    m.def("intersect_grad", py::overload_cast<const Line2<T>&, const Line2<T>&, const Point2<T>&, Line2<T>&, Line2<T>&>(&d3d::intersect_grad<T>),
        "Calculate gradient of intersect()");
    m.def("intersect_grad", [](const Quad2<T>& b1, const Quad2<T>& b2, const Poly2<T, 8>& grad, const vector<uint8_t>& xflags){
            Quad2<T> grad_p1, grad_p2;
            grad_p1.zero(); grad_p2.zero();
            intersect_grad(b1, b2, grad, xflags.data(), grad_p1, grad_p2);
            return make_tuple(grad_p1, grad_p2);
        }, "Calculate gradient of intersect()");
    m.def("intersect_grad", py::overload_cast<const AABox2<T>&, const AABox2<T>&, const AABox2<T>&, AABox2<T>&, AABox2<T>&>(&d3d::intersect_grad<T>),
        "Calculate gradient of intersect()");
    m.def("merge_grad", [](const Quad2<T>& b1, const Quad2<T>& b2, const Poly2<T, 8>& grad, const vector<uint8_t>& mflags){
            Quad2<T> grad_p1, grad_p2;
            grad_p1.zero(); grad_p2.zero();
            merge_grad(b1, b2, grad, mflags.data(), grad_p1, grad_p2);
            return make_tuple(grad_p1, grad_p2);
        }, "Calculate gradient of merge()");
    m.def("merge_grad", py::overload_cast<const AABox2<T>&, const AABox2<T>&, const AABox2<T>&, AABox2<T>&, AABox2<T>&>(&d3d::merge_grad<T>),
        "Calculate gradient of merge()");
    m.def("iou_grad", [](const Quad2<T>& b1, const Quad2<T>& b2, const T grad, const vector<uint8_t>& xflags){
            Quad2<T> grad_p1, grad_p2;
            grad_p1.zero(); grad_p2.zero();
            iou_grad(b1, b2, grad, xflags.size(), xflags.data(), grad_p1, grad_p2);
            return make_tuple(grad_p1, grad_p2);
        }, "Calculate gradient of iou()");
    m.def("iou_grad", py::overload_cast<const AABox2<T>&, const AABox2<T>&, const T&, AABox2<T>&, AABox2<T>&>(&d3d::iou_grad<T>),
        "Calculate gradient of iou()");
    m.def("giou_grad", [](const Quad2<T>& b1, const Quad2<T>& b2, const T grad,
        const vector<uint8_t>& xflags, const vector<uint8_t>& mflags){
            Quad2<T> grad_p1, grad_p2;
            grad_p1.zero(); grad_p2.zero();
            giou_grad(b1, b2, grad, xflags.size(), mflags.size(), xflags.data(), mflags.data(), grad_p1, grad_p2);
            return make_tuple(grad_p1, grad_p2);
        }, "Calculate gradient of giou()");
    m.def("giou_grad", py::overload_cast<const AABox2<T>&, const AABox2<T>&, const T&, AABox2<T>&, AABox2<T>&>(&d3d::giou_grad<T>),
        "Calculate gradient of giou()");
    m.def("diou_grad", [](const Quad2<T>& b1, const Quad2<T>& b2, const T grad,
    const vector<uint8_t>& xflags, const uint8_t& dflag1, const uint8_t& dflag2){
            Quad2<T> grad_p1, grad_p2;
            grad_p1.zero(); grad_p2.zero();
            diou_grad(b1, b2, grad, xflags.size(), dflag1, dflag2, xflags.data(), grad_p1, grad_p2);
            return make_tuple(grad_p1, grad_p2);
        }, "Calculate gradient of diou()");
    m.def("diou_grad", py::overload_cast<const AABox2<T>&, const AABox2<T>&, const T&, AABox2<T>&, AABox2<T>&>(&d3d::diou_grad<T>),
        "Calculate gradient of diou()");
}