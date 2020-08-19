
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <d3d/geometry.hpp>
#include <d3d/geometry_grad.hpp>

namespace py = pybind11;
using namespace std;
using namespace d3d;
using namespace pybind11::literals;

PYBIND11_MODULE(geometry, m) {
    m.doc() = "Python binding of the builtin geometry library of d3d, mainly for testing";

    py::class_<Point2d>(m, "Point2d")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("x", &Point2d::x)
        .def_readwrite("y", &Point2d::y);
    py::class_<Line2d>(m, "Line2d")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("a", &Line2d::a)
        .def_readwrite("b", &Line2d::b)
        .def_readwrite("c", &Line2d::c);
    py::class_<AABox2d>(m, "AABox2d")
        .def(py::init<>())
        .def(py::init<double, double, double, double>())
        .def_readwrite("min_x", &AABox2d::min_x)
        .def_readwrite("max_x", &AABox2d::max_x)
        .def_readwrite("min_y", &AABox2d::min_y)
        .def_readwrite("max_y", &AABox2d::max_y);
    py::class_<Box2d>(m, "Box2d")
        .def(py::init<>())
        .def_readonly("nvertices", &Box2d::nvertices)
        .def_property_readonly("vertices", [](const Box2d &b) {
            return vector<Point2d>(b.vertices, b.vertices + b.nvertices);});
    py::class_<Poly2d<8>>(m, "Poly2d8")
        .def(py::init<>())
        .def_readonly("nvertices", &Poly2d<8>::nvertices)
        .def_property_readonly("vertices", [](const Poly2d<8> &p) {
            return vector<Point2d>(p.vertices, p.vertices + p.nvertices);});

    m.def("line2_from_pp", &d3d::line2_from_pp<double>, "Create a line with two points");
    m.def("line2_from_xyxy", &d3d::line2_from_xyxy<double>, "Create a line with coordinate of two points");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<double, 4>, "Create bounding box of a polygon");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<double, 8>, "Create bounding box of a polygon");
    m.def("poly2_from_aabox2", &d3d::poly2_from_aabox2<double>, "Convert axis aligned box to polygon representation");
    m.def("poly2_from_xywhr", &d3d::poly2_from_xywhr<double>, "Creat a box with specified box parameters");

    m.def("area", py::overload_cast<const AABox2d&>(&d3d::area<double>), "Get the area of axis aligned box");
    m.def("area", py::overload_cast<const Box2d&>(&d3d::area<double, 4>), "Get the area of box");
    m.def("area", py::overload_cast<const Poly2d<8>&>(&d3d::area<double, 8>), "Get the area of polygon");
    m.def("distance", py::overload_cast<const Point2d&, const Point2d&>(&d3d::distance<double>),
        "Get the distance between two points");
    m.def("distance", py::overload_cast<const Line2d&, const Point2d&>(&d3d::distance<double>),
        "Get the distance from a point to a line");
    m.def("intersect", py::overload_cast<const Line2d&, const Line2d&>(&d3d::intersect<double>),
        "Get the intersection point of two lines");
    m.def("intersect", [](const Box2d& b1, const Box2d& b2){ return d3d::intersect(b1, b2); },
        "Get the intersection polygon of two boxes");
    m.def("intersect", [](const Box2d& b1, const Box2d& b2, vector<uint8_t> &xflags){
            uint8_t xflags_arr[8];
            auto result = d3d::intersect(b1, b2, xflags_arr);
            xflags.assign(xflags_arr, xflags_arr + result.nvertices);
            return result;
        }, "Get the intersection polygon of two boxes");
    m.def("intersect", py::overload_cast<const AABox2d&, const AABox2d&>(&d3d::intersect<double>),
        "Get the intersection box of two axis aligned boxes");
    m.def("iou", py::overload_cast<const AABox2d&, const AABox2d&>(&d3d::iou<double>),
        "Get the intersection over union of two axis aligned boxes");
    m.def("iou", [](const Box2d& b1, const Box2d& b2){ return d3d::iou(b1, b2); },
        "Get the intersection over union of two boxes");
    m.def("iou", [](const Box2d& b1, const Box2d& b2, vector<uint8_t> &xflags){
            uint8_t xflags_arr[8];
            auto result = d3d::iou(b1, b2, xflags_arr);
            xflags.assign(xflags_arr, xflags_arr + 8); // store all flags
            return result;
        }, "Get the intersection over union of two boxes");
    m.def("merge", [](const Box2d& b1, const Box2d& b2){ return d3d::merge(b1, b2); },
        "Get merged convex hull of two polygons");
    m.def("merge", py::overload_cast<const AABox2d&, const AABox2d&>(&d3d::merge<double>),
        "Get bounding box of two axis aligned boxes");
    m.def("max_distance", [](const Box2d& b1, const Box2d& b2){ uint8_t _; return d3d::max_distance(b1, b2, _, _);},
        "Get the max distance between two polygons");
    m.def("max_distance", py::overload_cast<const AABox2d&, const AABox2d&>(&d3d::max_distance<double>),
        "Get the max distance between two polygons");

    m.def("line2_from_pp_grad", &d3d::line2_from_pp_grad<double>, "Calculate gradient of line2_from_pp");
    m.def("line2_from_xyxy_grad", &d3d::line2_from_pp_grad<double>, "Calculate gradient of line2_from_xyxy");
    m.def("poly2_from_aabox2_grad", &d3d::poly2_from_aabox2_grad<double>, "Calculate gradient of poly2_from_aabox2");
    m.def("poly2_from_xywhr_grad", &d3d::poly2_from_xywhr_grad<double>, "Calculate gradient of poly2_from_xywhr");
    m.def("aabox2_from_poly2_grad", &d3d::aabox2_from_poly2_grad<double, 4>, "Calculate gradient of aabox2_from_poly2_grad");
    m.def("aabox2_from_poly2_grad", &d3d::aabox2_from_poly2_grad<double, 8>, "Calculate gradient of aabox2_from_poly2_grad");
    m.def("intersect_grad", py::overload_cast<const Line2d&, const Line2d&, const Point2d&, Line2d&, Line2d&>(&d3d::intersect_grad<double>),
        "Calculate gradient of intersect");
}