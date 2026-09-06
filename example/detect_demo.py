# VexelLibFaceDetection - Python Detection & Visualization Demo
# Copyright (c) 2026, Vexel Innovations. All rights reserved.

import cv2
import numpy as np
import pyfacedetect
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="VexelLibFaceDetection Python Demo — Vexel Innovations")
    parser.add_argument("--image", type=str, default="images/cnnresult.png", help="Path to input image")
    parser.add_argument("--iterations", type=int, default=1, help="Number of benchmark iterations")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    detector = pyfacedetect.FaceDetector()
    options = pyfacedetect.DetectionOptions()
    options.confidence_threshold = 0.3

    print("==============================================")
    print("   VexelLibFaceDetection  |  Vexel Innovations")
    print("==============================================")
    print(f"Image: {args.image} — shape: {image.shape}")

    # Benchmark
    start = time.perf_counter()
    for _ in range(args.iterations):
        faces = detector.detect(image, options)
    end = time.perf_counter()

    avg_ms = ((end - start) / args.iterations) * 1000.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"\n--- Detection Results ---")
    print(f"Faces Detected : {len(faces)}")
    print(f"Avg Latency    : {avg_ms:.2f} ms")
    print(f"Throughput     : {fps:.2f} FPS\n")

    for i, face in enumerate(faces):
        print(f"[Face {i+1}] score={face.score:.4f} {face.bbox}")
        x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        for k, lm in enumerate(face.landmarks):
            cv2.circle(image, (lm.x, lm.y), 3, colors[k], -1)

    output_path = "vexel_detected_output.png"
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
