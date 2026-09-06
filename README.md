# VexelLibFaceDetection & AI Predictive Vision Suite

[![Vexel Innovations](https://img.shields.io/badge/Vexel-Innovations-00D2FF.svg)](https://github.com/Vexel-Innovations)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![SIMD Acceleration](https://img.shields.io/badge/SIMD-AVX2-green.svg)](#features)
[![License](https://img.shields.io/badge/License-3--Clause%20BSD-orange.svg)](LICENSE)

An open-source, ultra-high-performance **C++20 & AVX2 Neural Vision Engine** with a **Multi-Modal AI Predictive Analytics & Action Recognition Suite**, developed and maintained by **Vexel Innovations**.

---

## ✨ Features

- **⚡ Modern C++20 Engine Core**: RAII wrapper (`vexelfacedetect::FaceDetector`), `std::span` zero-copy image views, type-safe API.
- **🚀 SIMD Accelerated**: Optimized with Intel AVX2 instruction sets for real-time high-throughput face & landmark inference.
- **🎯 5-Point Landmark Detection**: Calculates exact coordinates for left eye, right eye, nose tip, left mouth corner, right mouth corner.
- **🔍 Multi-Class Object Identification & Labeling**: Detects and labels **Persons**, **Cell Phones / Devices**, **Laptops / Monitors**, **Bottles / Cups**, and **Bags/Items** in live video feeds.
- **🏃 Human Action State Analysis**: Real-time action recognition (`Standing`, `Walking Forward`, `Traversing`, `Sitting / Idle`).
- **🔮 Predictive Trajectory Analytics**: Calculates velocity vectors ($v_x, v_y$) and predicts future motion paths with visual vector arrows (`PREDICT: Approaching Camera`, `Moving Away`).
- **🖥️ Real-Time Camera HUD**: Live dashboard displaying translucent performance analytics (FPS, Latency ms, Object Count, Face Count).

---

## 🚀 Quick Start

### 1. Build C++20 Benchmark CLI
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### 2. Run Image Face Benchmark
```bash
./vexel_detect_cli images/cnnresult.png 1
```

### 3. Run Real-Time AI Camera & Object Detection HUD
```bash
python example/vexel_predictive_vision_demo.py
```

---

## 💡 C++20 API Usage

```cpp
#include "vexelfacedetect.hpp"
#include <iostream>
#include <vector>

int main() {
    // 1. Instantiate Vexel Detector
    vexelfacedetect::FaceDetector detector;

    // 2. Configure Options
    vexelfacedetect::DetectionOptions options;
    options.confidence_threshold = 0.4f;

    // 3. Perform Inference on Image Span (BGR)
    std::vector<uint8_t> image_bgr = /* load BGR image */;
    int width = 1280, height = 960, step = width * 3;

    auto faces = detector.detect(image_bgr, width, height, step, options);

    // 4. Iterate Results
    for (const auto& face : faces) {
        std::cout << "Face found! Score: " << face.score 
                  << " BBox: (" << face.bbox.x << ", " << face.bbox.y << ")\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << " Landmark " << i+1 << ": (" 
                      << face.landmarks[i].x << ", " << face.landmarks[i].y << ")\n";
        }
    }
    return 0;
}
```

---

## 👥 Contributors & Team

Crafted with ❤️ by the **Vexel Innovations** core engineering team:

- **[Usama (Uszkido)](https://github.com/Uszkido)**
- **[Vexelpro](https://github.com/Vexelpro)**
- **[Vexel Innovations](https://github.com/Vexel-Innovations)**

---

## 📜 License

This project is licensed under the **3-Clause BSD License**. See [LICENSE](LICENSE) for details.
