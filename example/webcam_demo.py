"""
VexelLibFaceDetection - High-Performance Real-Time Webcam Face Scanner
Copyright (c) 2026, Vexel Innovations. All rights reserved.
Uses VexelLibFaceDetection C++20 AVX2 CNN Engine via ctypes.
"""

import cv2
import time
import argparse
import ctypes
import os
import numpy as np

# Buffer constants matching C++ facedetectcnn.h
FACEDETECTION_RESULT_BUFFER_SIZE = 0x9000
FACEDETECTION_RESULT_STRIDE_SHORTS = 16

class VexelCPPDetector:
    def __init__(self, dll_path):
        self.dll_path = os.path.abspath(dll_path)
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"DLL not found at: {self.dll_path}")

        # Ensure MinGW bin directory is in DLL search path
        mingw_bin = r"C:\msys64\mingw64\bin"
        if os.path.exists(mingw_bin) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(mingw_bin)
            os.add_dll_directory(os.path.dirname(self.dll_path))

        self.lib = ctypes.CDLL(self.dll_path)

        # int * facedetect_cnn(unsigned char * result_buffer, unsigned char * rgb_image_data, int width, int height, int step)
        self.lib.facedetect_cnn.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.facedetect_cnn.restype = ctypes.POINTER(ctypes.c_int)

        # Create aligned 64-byte result buffer
        self.buffer = (ctypes.c_ubyte * FACEDETECTION_RESULT_BUFFER_SIZE)()

    def detect(self, bgr_image, confidence_threshold=0.3):
        h, w, c = bgr_image.shape
        step = bgr_image.strides[0]

        img_ptr = bgr_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        buf_ptr = ctypes.cast(self.buffer, ctypes.POINTER(ctypes.c_ubyte))

        p_result = self.lib.facedetect_cnn(buf_ptr, img_ptr, w, h, step)
        if not p_result:
            return []

        count = p_result[0]
        if count <= 0:
            return []

        # Convert pointer to integer memory address then cast to int16_t array pointer
        p_result_addr = ctypes.addressof(p_result.contents)
        p_shorts_addr = p_result_addr + ctypes.sizeof(ctypes.c_int)
        p_shorts = ctypes.cast(p_shorts_addr, ctypes.POINTER(ctypes.c_int16))

        faces = []
        for i in range(count):
            offset = i * FACEDETECTION_RESULT_STRIDE_SHORTS
            confidence = p_shorts[offset + 0]
            score = confidence / 100.0

            if score < confidence_threshold:
                continue

            x = int(p_shorts[offset + 1])
            y = int(p_shorts[offset + 2])
            w_box = int(p_shorts[offset + 3])
            h_box = int(p_shorts[offset + 4])

            landmarks = []
            for j in range(5):
                lx = int(p_shorts[offset + 5 + j * 2])
                ly = int(p_shorts[offset + 5 + j * 2 + 1])
                landmarks.append((lx, ly))

            faces.append({
                'score': score,
                'bbox': (x, y, w_box, h_box),
                'landmarks': landmarks
            })

        return faces

def run_live_webcam(camera_id=0):
    print("==========================================================")
    print(" VexelLibFaceDetection | Real-Time C++20 AVX2 Camera Scanner")
    print(" Powered by Vexel Innovations Core CNN Engine            ")
    print("==========================================================")

    dll_path = os.path.join(os.path.dirname(__file__), "..", "build", "vexelfacedetection.dll")
    try:
        detector = VexelCPPDetector(dll_path)
        print(f"[Vexel Engine] Successfully loaded C++20 Engine from: {dll_path}")
    except Exception as e:
        print(f"[Vexel Engine Error] Failed to load C++ DLL: {e}")
        return

    print(f"Opening camera ID: {camera_id}...")
    print("Press 'q' or 'ESC' on the camera window to exit.\n")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Unable to access camera device {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    print("[Vexel Engine] Camera live feed started!")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        t0 = time.time()
        
        # Ensure BGR array is contiguous
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # High accuracy CNN detect using Vexel C++ engine
        faces = detector.detect(frame, confidence_threshold=0.35)

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        fps = 1.0 / (t1 - prev_time + 1e-6)
        prev_time = t1

        h, w_img, _ = frame.shape

        # Render Vexel Innovations Bounding Box (Teal/Cyan: #00D2FF) & 5 Landmarks
        for face in faces:
            x, y, w_box, h_box = face['bbox']
            score = face['score']
            landmarks = face['landmarks']

            # Draw Bounding Box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 210, 0), 2)

            # Score badge
            cv2.rectangle(frame, (x, max(0, y - 22)), (x + 130, max(0, y)), (255, 210, 0), -1)
            cv2.putText(frame, f"Vexel CNN: {score:.2f}", (x + 4, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw 5 Facial Landmarks (Red dots)
            for lm in landmarks:
                cv2.circle(frame, lm, 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Draw HUD Header Banner
        cv2.rectangle(frame, (0, 0), (w_img, 35), (15, 15, 15), -1)
        cv2.putText(frame, "VEXEL LIB FACE DETECTION | REAL-TIME C++20 AVX2 SCANNER", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 255), 2, cv2.LINE_AA)

        # Draw HUD Stats Footer
        stats_text = f"FPS: {fps:.1f} | CNN Latency: {latency_ms:.1f} ms | Detected Faces: {len(faces)} | Vexel Innovations"
        cv2.rectangle(frame, (0, h - 30), (w_img, h), (15, 15, 15), -1)
        cv2.putText(frame, stats_text, (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("VexelLibFaceDetection - Real-Time Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Vexel Engine] Live scanning ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VexelLibFaceDetection Live Webcam Scanner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()
    run_live_webcam(args.camera)
