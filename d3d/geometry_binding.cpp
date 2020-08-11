
#include <pybind11/pybind11.h>
#include <d3d/geometry.hpp>

namespace py = pybind11;
using namespace d3d;
using namespace pybind11::literals;

PYBIND11_MODULE(geometry, m) {
    m.doc() = "Python binding of the builtin geometry library of d3d, mainly for testing";

    py::class_<Point2f>(m, "Point2f")
        .def(py::init<>())
        .def(py::init<float, float>())
        .def_readwrite("x", &Point2f::x)
        .def_readwrite("y", &Point2f::y);
    py::class_<Line2f>(m, "Line2f")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("a", &Line2f::a)
        .def_readwrite("b", &Line2f::b)
        .def_readwrite("c", &Line2f::c);
    py::class_<AABox2f>(m, "AABox2f")
        .def(py::init<>())
        .def(py::init<float, float, float, float>())
        .def_readwrite("min_x", &AABox2f::min_x)
        .def_readwrite("max_x", &AABox2f::max_x)
        .def_readwrite("min_y", &AABox2f::min_y)
        .def_readwrite("max_y", &AABox2f::max_y);
    py::class_<Box2f>(m, "Box2f")
        .def(py::init<>());
    py::class_<Poly2f<8>>(m, "Poly2f8")
        .def(py::init<>())
        .def_readonly("nvertices", &Poly2f<8>::nvertices)
        .def_readonly("vertices", &Poly2f<8>::vertices); // TODO: not correct

    m.def("area", py::overload_cast<const AABox2f&>(&d3d::area<float>), "Get the area of axis aligned box");
    m.def("area", py::overload_cast<const Box2f&>(&d3d::area<float, 4>), "Get the area of box");
    m.def("area", py::overload_cast<const Poly2f<8>&>(&d3d::area<float, 8>), "Get the area of polygon");
    m.def("distance", py::overload_cast<const Line2f&, const Point2f&>(&d3d::distance<float>),
        "Get the distance from a point to a line");
    m.def("intersect", py::overload_cast<const Line2f&, const Line2f&>(&d3d::intersect<float>),
        "Get the intersection point of two lines");
    m.def("intersect", py::overload_cast<const Box2f&, const Box2f&>(&d3d::intersect<float, 4, 4>),
        "Get the intersection polygon of two boxes");
    m.def("intersect", py::overload_cast<const AABox2f&, const AABox2f&>(&d3d::intersect<float>),
        "Get the intersection box of two axis aligned boxes");
    m.def("iou", py::overload_cast<const AABox2f&, const AABox2f&>(&d3d::iou<float>),
        "Get the intersection over union of two axis aligned boxes");

    m.def("line2_from_pp", &d3d::line2_from_pp<float>, "Create a line with two points");
    m.def("line2_from_xyxy", &d3d::line2_from_xyxy<float>, "Create a line with coordinate of two points");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<float, 4>, "Create bounding box of a polygon");
    m.def("aabox2_from_poly2", &d3d::aabox2_from_poly2<float, 8>, "Create bounding box of a polygon");
    m.def("poly2_from_aabox2", &d3d::poly2_from_aabox2<float>, "Convert axis aligned box to polygon representation");
    m.def("poly2_from_xywhr", &d3d::poly2_from_xywhr<float>, "Creat a box with specified box parameters");
}