# VexelLibFaceDetection

<p align="center">
  <strong>High-Performance CNN-Based Face Detection Engine</strong><br>
  <em>Engineered by Vexel Innovations</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square" />
  <img src="https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX512%20%7C%20NEON-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-lightgrey?style=flat-square" />
</p>

---

VexelLibFaceDetection is a modernized, production-grade CNN face detection library built on C++20 abstractions and high-performance SIMD acceleration. The CNN model weights are embedded as static arrays in C++ source — no external model files or runtime dependencies required. Supports Windows, Linux, macOS, and ARM platforms.

A native **Python extension** (`pyfacedetect`) is included for seamless NumPy/OpenCV integration.

![Detection Example](/images/cnnresult.png "VexelLibFaceDetection Detection Result")

## Features

- **Zero External Dependencies** — self-contained C++ source, only needs a C++20 compiler
- **Modern C++20 API** — type-safe `vexelfacedetect::FaceDetector` class with `std::span`, `std::array`, RAII
- **SIMD Acceleration** — AVX2, AVX512, and ARM NEON for maximum throughput
- **5-Point Facial Landmarks** — eyes, nose tip, and mouth corners per detected face
- **Python Bindings** — Pybind11-powered `pyfacedetect` package with zero-copy NumPy array input
- **Cross-Platform** — Windows (MSVC), Linux (GCC/Clang), macOS, ARM (Raspberry Pi)
- **OpenMP Multi-Threading** — automatic parallelism for batch and real-time workloads

## Quick Start

### C++ Usage

```cpp
#include "vexelfacedetect.hpp"

vexelfacedetect::FaceDetector detector;
auto faces = detector.detect(image_span, width, height);

for (const auto& face : faces) {
    printf("score=%.2f bbox=[%d,%d,%d,%d]\n",
           face.score, face.bbox.x, face.bbox.y,
           face.bbox.width, face.bbox.height);
}
```

### Python Usage

```python
import pyfacedetect
import cv2

image = cv2.imread("photo.jpg")
detector = pyfacedetect.FaceDetector()
faces = detector.detect(image)

for face in faces:
    print(f"score={face.score:.2f} {face.bbox}")
```

## Building from Source

### Prerequisites

- C++20 compatible compiler (MSVC 2019+, GCC 10+, Clang 11+)
- CMake 3.16+
- (Optional) OpenCV for demo executables
- (Optional) Python 3.10+ with `pybind11` for Python bindings

### C++ Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

Build options:

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_AVX2` | ON | Enable AVX2 SIMD instructions (x86) |
| `ENABLE_AVX512` | OFF | Enable AVX512 SIMD instructions (x86) |
| `ENABLE_NEON` | OFF | Enable NEON SIMD instructions (ARM) |
| `USE_OPENMP` | ON | Enable OpenMP multi-threading |
| `DEMO` | OFF | Build OpenCV demo executables |

### Python Build

```bash
pip install pybind11
python setup.py build_ext --inplace
# or
pip install -e .
```

## Performance

### Intel CPU (AVX2)

| Resolution | Single-Thread | Multi-Thread (16T) |
|------------|---------------|---------------------|
| 640×480 | 50.02 ms / 19.99 FPS | 6.55 ms / 152.65 FPS |
| 320×240 | 13.09 ms / 76.39 FPS | 1.82 ms / 550.54 FPS |
| 160×120 | 3.61 ms / 277.37 FPS | 0.57 ms / 1745.13 FPS |
| 128×96 | 2.11 ms / 474.60 FPS | 0.33 ms / 2994.23 FPS |

* Intel Core i7-7820X @ 3.60GHz, 16 threads

### ARM Linux (Raspberry Pi 4 B)

| Resolution | Single-Thread | Multi-Thread (4T) |
|------------|---------------|---------------------|
| 640×480 | 404.63 ms / 2.47 FPS | 125.47 ms / 7.97 FPS |
| 320×240 | 105.73 ms / 9.46 FPS | 32.98 ms / 30.32 FPS |
| 160×120 | 26.05 ms / 38.38 FPS | 7.91 ms / 126.49 FPS |
| 128×96 | 15.06 ms / 66.38 FPS | 4.50 ms / 222.28 FPS |

* Raspberry Pi 4 B, Cortex-A72 (ARMv8) @ 1.5GHz

### WIDER Face Benchmark

```
AP_easy=0.887, AP_medium=0.871, AP_hard=0.768
```

## Project Structure

```
vexelibfacedetection/
├── src/
│   ├── vexelfacedetect.hpp          # Modern C++20 public API
│   ├── vexelfacedetect.cpp          # Modern API implementation
│   ├── facedetectcnn.h              # Core CNN engine header
│   ├── facedetectcnn.cpp            # Core CNN inference engine
│   ├── facedetectcnn-model.cpp      # Network architecture
│   └── facedetectcnn-data.cpp       # Embedded model weights
├── python/
│   └── pyfacedetect.cpp             # Pybind11 Python bindings
├── example/
│   ├── modern_detect_cli.cpp        # C++20 CLI benchmark tool
│   ├── detect_demo.py               # Python detection demo
│   ├── detect-image.cpp             # OpenCV image detection
│   └── detect-camera.cpp            # OpenCV camera detection
├── setup.py                         # Python package build
├── CMakeLists.txt                   # CMake build system
└── LICENSE
```

## Examples

### Modern CLI Benchmark

```bash
./modern_detect_cli images/cnnresult.png 100
```

### Python Detection with Visualization

```bash
python example/detect_demo.py --image images/cnnresult.png
```

## Organization

**Vexel Innovations** — Engineering high-performance computer vision and AI infrastructure.

### Core Team & Contributors

| Contributor / Entity | Role / Function | GitHub Profile |
|----------------------|-----------------|----------------|
| **Vexel Innovations** | Lead Organization & AI Infrastructure | [@Vexel-Innovations](https://github.com/Vexel-Innovations) |
| **Uszkido / Usama** | Lead Architect & Modernization Engineer | [@Uszkido](https://github.com/Uszkido) |
| **Vexelpro** | Core Contributor & Systems Verification | [@Vexelpro](https://github.com/Vexelpro) |

### Acknowledgments

This project is built upon foundational research in CNN-based face detection. VexelLibFaceDetection extends and modernizes the original work with C++20 architecture, Python bindings, and production-grade engineering by the Vexel Innovations team.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{vexelibfacedetection2026,
    title     = {VexelLibFaceDetection: High-Performance CNN Face Detection Engine},
    author    = {Vexel Innovations},
    year      = {2026},
    url       = {https://github.com/Vexel-Innovations/libfacedetection}
}
```
