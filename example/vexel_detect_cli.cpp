/*
 VexelLibFaceDetection - C++20 CLI Benchmark & Inference Tool
 Copyright (c) 2026, Vexel Innovations. All rights reserved.
 Licensed under the BSD 3-Clause License.
*/

#include "vexelfacedetect.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "VexelLibFaceDetection CLI v1.0.0\n";
        std::cout << "Usage: vexel_detect_cli <image_path> [iterations=1]\n";
        return 1;
    }

    std::string image_path = argv[1];
    int iterations = (argc >= 3) ? std::stoi(argv[2]) : 1;

    int width = 0, height = 0, channels = 0;
    unsigned char* img_raw = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!img_raw) {
        std::cerr << "Error: Failed to load image " << image_path << "\n";
        return 1;
    }

    std::cout << "==============================================\n";
    std::cout << "   VexelLibFaceDetection  |  Vexel Innovations\n";
    std::cout << "==============================================\n";
    std::cout << "Image: " << image_path << " (" << width << "x" << height << ", " << channels << " ch)\n";

    // Convert RGB -> BGR
    for (int i = 0; i < width * height; ++i) {
        std::swap(img_raw[i * 3 + 0], img_raw[i * 3 + 2]);
    }

    std::span<const uint8_t> image_span(img_raw, width * height * 3);
    vexelfacedetect::FaceDetector detector;

    std::vector<vexelfacedetect::Face> faces;

    // Warmup
    faces = detector.detect(image_span, width, height);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        faces = detector.detect(image_span, width, height);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double fps = 1000.0 / avg_ms;

    std::cout << "\n--- Detection Results ---\n";
    std::cout << "Faces Detected : " << faces.size() << "\n";
    std::cout << "Avg Latency    : " << avg_ms << " ms\n";
    std::cout << "Throughput     : " << fps << " FPS\n\n";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        std::cout << "[Face " << (i + 1) << "] score=" << f.score
                  << "  bbox=[" << f.bbox.x << "," << f.bbox.y
                  << "," << f.bbox.width << "," << f.bbox.height << "]\n";
        for (int k = 0; k < 5; ++k) {
            std::cout << "  Landmark " << k + 1 << ": (" << f.landmarks[k].x << ", " << f.landmarks[k].y << ")\n";
        }
    }

    stbi_image_free(img_raw);
    return 0;
}
