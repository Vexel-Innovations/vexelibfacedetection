/*
 VexelLibFaceDetection Modern C++20 Interface
 Copyright (c) Vexel Innovations
*/

#pragma once

#if __has_include("vexelfacedetection_export.h")
#include "vexelfacedetection_export.h"
#elif __has_include("facedetection_export.h")
#include "facedetection_export.h"
#else
#define FACEDETECTION_EXPORT
#endif

#include <vector>
#include <span>
#include <cstdint>
#include <string>
#include <memory>
#include <array>
#include <optional>

namespace vexelfacedetect {

struct Point2D {
    int x{0};
    int y{0};
};

struct BoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};
};

struct Face {
    float score{0.0f};
    BoundingBox bbox{};
    std::array<Point2D, 5> landmarks{}; // 5 facial landmarks
};

struct DetectionOptions {
    float confidence_threshold{0.3f};
    float nms_threshold{0.45f};
    int top_k{1000};
    int keep_top_k{750};
};

class FACEDETECTION_EXPORT FaceDetector {
public:
    FaceDetector() = default;
    ~FaceDetector() = default;

    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;
    FaceDetector(FaceDetector&&) noexcept = default;
    FaceDetector& operator=(FaceDetector&&) noexcept = default;

    // Detect faces in BGR pixel array with span
    [[nodiscard]] std::vector<Face> detect(
        std::span<const uint8_t> image_data,
        int width,
        int height,
        int step,
        const DetectionOptions& options = {}
    ) const;

    // Helper for packed BGR buffer where step = width * 3
    [[nodiscard]] std::vector<Face> detect(
        std::span<const uint8_t> image_data,
        int width,
        int height,
        const DetectionOptions& options = {}
    ) const {
        return detect(image_data, width, height, width * 3, options);
    }
};

} // namespace vexelfacedetect
