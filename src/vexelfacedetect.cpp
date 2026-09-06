/*
 VexelLibFaceDetection - Modern C++20 API Implementation
 Copyright (c) 2026, Vexel Innovations. All rights reserved.
 Licensed under the BSD 3-Clause License.
*/

#include "vexelfacedetect.hpp"
#include "facedetectcnn.h"
#include <vector>

namespace vexelfacedetect {

std::vector<Face> FaceDetector::detect(
    std::span<const uint8_t> image_data,
    int width,
    int height,
    int step,
    const DetectionOptions& options
) const {
    std::vector<Face> results;

    if (image_data.empty() || width <= 0 || height <= 0) {
        return results;
    }

    // Call core CNN engine
    std::vector<FaceRect> faces = objectdetect_cnn(
        image_data.data(),
        width,
        height,
        step
    );

    results.reserve(faces.size());
    for (const auto& f : faces) {
        if (f.score < options.confidence_threshold) {
            continue;
        }

        Face face{};
        face.score = f.score;
        face.bbox = BoundingBox{
            .x = f.x,
            .y = f.y,
            .width = f.w,
            .height = f.h
        };

        for (int i = 0; i < 5; ++i) {
            face.landmarks[i] = Point2D{
                .x = f.lm[i * 2],
                .y = f.lm[i * 2 + 1]
            };
        }

        results.push_back(face);
    }

    return results;
}

} // namespace vexelfacedetect
