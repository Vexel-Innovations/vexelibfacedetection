# VexelLibFaceDetection - Python Pybind11 Extension
# Copyright (c) 2026, Vexel Innovations. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../src/vexelfacedetect.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyfacedetect, m) {
    m.doc() = "VexelLibFaceDetection — High-Performance Face Detection by Vexel Innovations";

    py::class_<vexelfacedetect::Point2D>(m, "Point2D")
        .def_readwrite("x", &vexelfacedetect::Point2D::x)
        .def_readwrite("y", &vexelfacedetect::Point2D::y)
        .def("__repr__", [](const vexelfacedetect::Point2D& p) {
            return "(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
        });

    py::class_<vexelfacedetect::BoundingBox>(m, "BoundingBox")
        .def_readwrite("x", &vexelfacedetect::BoundingBox::x)
        .def_readwrite("y", &vexelfacedetect::BoundingBox::y)
        .def_readwrite("width", &vexelfacedetect::BoundingBox::width)
        .def_readwrite("height", &vexelfacedetect::BoundingBox::height)
        .def("__repr__", [](const vexelfacedetect::BoundingBox& b) {
            return "[x=" + std::to_string(b.x) + ", y=" + std::to_string(b.y) + 
                   ", w=" + std::to_string(b.width) + ", h=" + std::to_string(b.height) + "]";
        });

    py::class_<vexelfacedetect::Face>(m, "Face")
        .def_readwrite("score", &vexelfacedetect::Face::score)
        .def_readwrite("bbox", &vexelfacedetect::Face::bbox)
        .def_readwrite("landmarks", &vexelfacedetect::Face::landmarks);

    py::class_<vexelfacedetect::DetectionOptions>(m, "DetectionOptions")
        .def(py::init<>())
        .def_readwrite("confidence_threshold", &vexelfacedetect::DetectionOptions::confidence_threshold)
        .def_readwrite("nms_threshold", &vexelfacedetect::DetectionOptions::nms_threshold)
        .def_readwrite("top_k", &vexelfacedetect::DetectionOptions::top_k)
        .def_readwrite("keep_top_k", &vexelfacedetect::DetectionOptions::keep_top_k);

    py::class_<vexelfacedetect::FaceDetector>(m, "FaceDetector")
        .def(py::init<>())
        .def("detect", [](const vexelfacedetect::FaceDetector& self, 
                          py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
                          const vexelfacedetect::DetectionOptions& options) {
            py::buffer_info buf = image.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::invalid_argument("Input image must have shape (height, width, 3) BGR or RGB");
            }

            int height = static_cast<int>(buf.shape[0]);
            int width = static_cast<int>(buf.shape[1]);
            int step = static_cast<int>(buf.strides[0]);

            const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
            std::span<const uint8_t> image_span(ptr, buf.size);

            return self.detect(image_span, width, height, step, options);
        }, py::arg("image"), py::arg("options") = vexelfacedetect::DetectionOptions{});
}
