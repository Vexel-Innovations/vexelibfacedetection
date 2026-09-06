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

    alignas(64) uint8_t result_buffer[FACEDETECTION_RESULT_BUFFER_SIZE];
    
    int* pResults = facedetect_cnn(
        result_buffer,
        const_cast<unsigned char*>(image_data.data()),
        width,
        height,
        step
    );

    if (!pResults) {
        return results;
    }

    int count = pResults[0];
    results.reserve(count);

    for (int i = 0; i < count; ++i) {
        int16_t* p = (int16_t*)(pResults + 1) + i * FACEDETECTION_RESULT_STRIDE_SHORTS;
        int confidence = p[0];
        int x = p[1];
        int y = p[2];
        int w = p[3];
        int h = p[4];

        if (confidence < options.confidence_threshold * 100) {
            continue;
        }

        Face face{};
        face.score = static_cast<float>(confidence) / 100.0f;
        face.bbox = BoundingBox{
            .x = x,
            .y = y,
            .width = w,
            .height = h
        };

        for (int j = 0; j < 5; ++j) {
            face.landmarks[j] = Point2D{
                .x = static_cast<int>(p[5 + j * 2]),
                .y = static_cast<int>(p[5 + j * 2 + 1])
            };
        }

        results.push_back(face);
    }
    return results;
}

} // namespace vexelfacedetect
