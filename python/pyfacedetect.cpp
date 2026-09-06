#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../src/facedetect20.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyfacedetect, m) {
    m.doc() = "Modern C++20 High-Performance Face Detection Python Extension";

    py::class_<facedetect::Point2D>(m, "Point2D")
        .def_readwrite("x", &facedetect::Point2D::x)
        .def_readwrite("y", &facedetect::Point2D::y)
        .def("__repr__", [](const facedetect::Point2D& p) {
            return "(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
        });

    py::class_<facedetect::BoundingBox>(m, "BoundingBox")
        .def_readwrite("x", &facedetect::BoundingBox::x)
        .def_readwrite("y", &facedetect::BoundingBox::y)
        .def_readwrite("width", &facedetect::BoundingBox::width)
        .def_readwrite("height", &facedetect::BoundingBox::height)
        .def("__repr__", [](const facedetect::BoundingBox& b) {
            return "[x=" + std::to_string(b.x) + ", y=" + std::to_string(b.y) + 
                   ", w=" + std::to_string(b.width) + ", h=" + std::to_string(b.height) + "]";
        });

    py::class_<facedetect::Face>(m, "Face")
        .def_readwrite("score", &facedetect::Face::score)
        .def_readwrite("bbox", &facedetect::Face::bbox)
        .def_readwrite("landmarks", &facedetect::Face::landmarks);

    py::class_<facedetect::DetectionOptions>(m, "DetectionOptions")
        .def(py::init<>())
        .def_readwrite("confidence_threshold", &facedetect::DetectionOptions::confidence_threshold)
        .def_readwrite("nms_threshold", &facedetect::DetectionOptions::nms_threshold)
        .def_readwrite("top_k", &facedetect::DetectionOptions::top_k)
        .def_readwrite("keep_top_k", &facedetect::DetectionOptions::keep_top_k);

    py::class_<facedetect::FaceDetector>(m, "FaceDetector")
        .def(py::init<>())
        .def("detect", [](const facedetect::FaceDetector& self, 
                          py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
                          const facedetect::DetectionOptions& options) {
            py::buffer_info buf = image.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw std::invalid_argument("Input image array must have shape (height, width, 3) BGR or RGB");
            }

            int height = static_cast<int>(buf.shape[0]);
            int width = static_cast<int>(buf.shape[1]);
            int step = static_cast<int>(buf.strides[0]);

            const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
            std::span<const uint8_t> image_span(ptr, buf.size);

            return self.detect(image_span, width, height, step, options);
        }, py::arg("image"), py::arg("options") = facedetect::DetectionOptions{});
}
