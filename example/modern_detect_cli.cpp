#include "facedetect20.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "../example/stb_image.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: modern_detect_cli <image_path> [iterations=100]\n";
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

    std::cout << "===========================================\n";
    std::cout << "      libfacedetection C++20 Modern CLI    \n";
    std::cout << "===========================================\n";
    std::cout << "Image loaded: " << image_path << " (" << width << "x" << height << ", " << channels << " ch)\n";

    // Convert RGB to BGR in-place as expected byfacedetect_cnn
    for (int i = 0; i < width * height; ++i) {
        std::swap(img_raw[i * 3 + 0], img_raw[i * 3 + 2]);
    }

    std::span<const uint8_t> image_span(img_raw, width * height * 3);
    facedetect::FaceDetector detector;

    std::vector<facedetect::Face> faces;
    
    // Warmup
    faces = detector.detect(image_span, width, height);

    // Benchmark loop
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        faces = detector.detect(image_span, width, height);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_ms = total_ms / iterations;
    double fps = 1000.0 / avg_ms;

    std::cout << "\nResults:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Detected Faces : " << faces.size() << "\n";
    std::cout << "Avg Latency    : " << avg_ms << " ms\n";
    std::cout << "Throughput     : " << fps << " FPS\n";
    std::cout << "-------------------------------------------\n\n";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        std::cout << "Face #" << (i + 1) << ": score=" << f.score
                  << " [x=" << f.bbox.x << ", y=" << f.bbox.y
                  << ", w=" << f.bbox.width << ", h=" << f.bbox.height << "]\n";
        std::cout << "  Landmarks (5 points):\n";
        for (int k = 0; k < 5; ++k) {
            std::cout << "    L" << k + 1 << ": (" << f.landmarks[k].x << ", " << f.landmarks[k].y << ")\n";
        }
    }

    stbi_image_free(img_raw);
    return 0;
}
