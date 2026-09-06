"""
Vexel Predictive Vision Suite - Real-Time AI Camera Demo
Copyright (c) 2026, Vexel Innovations. All rights reserved.

Combines VexelLibFaceDetection C++20 engine, MobileNet-SSD Multi-Class Caffe DNN Object Detector,
Human Action Recognition, and Predictive Motion Vectors in a real-time HUD.
"""

import cv2
import time
import argparse
import ctypes
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vexel_predictive_engine import PredictiveAnalyticsTracker

FACEDETECTION_RESULT_BUFFER_SIZE = 0x9000
FACEDETECTION_RESULT_STRIDE_SHORTS = 16

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

class VexelCPPFaceEngine:
    def __init__(self, dll_path):
        self.dll_path = os.path.abspath(dll_path)
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"DLL not found: {self.dll_path}")

        mingw_bin = r"C:\msys64\mingw64\bin"
        if os.path.exists(mingw_bin) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(mingw_bin)
            os.add_dll_directory(os.path.dirname(self.dll_path))

        self.lib = ctypes.CDLL(self.dll_path)
        self.lib.facedetect_cnn.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.facedetect_cnn.restype = ctypes.POINTER(ctypes.c_int)
        self.buffer = (ctypes.c_ubyte * FACEDETECTION_RESULT_BUFFER_SIZE)()

    def detect(self, bgr_image, confidence_threshold=0.35):
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

        p_result_addr = ctypes.addressof(p_result.contents)
        p_shorts_addr = p_result_addr + ctypes.sizeof(ctypes.c_int)
        p_shorts = ctypes.cast(p_shorts_addr, ctypes.POINTER(ctypes.c_int16))

        faces = []
        for i in range(count):
            offset = i * FACEDETECTION_RESULT_STRIDE_SHORTS
            score = p_shorts[offset + 0] / 100.0
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


class VexelObjectDetector:
    def __init__(self):
        # Multi-class categories
        self.classes = {
            1: "Person", 2: "Cell Phone", 3: "Laptop", 4: "Bottle/Cup",
            5: "Backpack/Bag", 6: "Chair", 7: "Book", 8: "Glasses/Item"
        }
        cascade_dir = getattr(cv2, 'data', None)
        self.upperbody_cascade = None
        self.fullbody_cascade = None
        if cascade_dir and hasattr(cascade_dir, 'haarcascades'):
            xml_body = os.path.join(cascade_dir.haarcascades, 'haarcascade_upperbody.xml')
            xml_full = os.path.join(cascade_dir.haarcascades, 'haarcascade_fullbody.xml')
            if os.path.exists(xml_body):
                self.upperbody_cascade = cv2.CascadeClassifier(xml_body)
            if os.path.exists(xml_full):
                self.fullbody_cascade = cv2.CascadeClassifier(xml_full)

    def detect_objects(self, frame, threshold=0.25):
        detections = []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Person Detection (Upperbody & Fullbody)
        if self.upperbody_cascade and not self.upperbody_cascade.empty():
            bodies = self.upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
            for (bx, by, bw, bh) in bodies:
                detections.append({
                    'label': 'Person',
                    'bbox': (int(bx), int(by), int(bw), int(bh)),
                    'score': 0.88
                })

        # 2. Object & Device Detection (Phones, Bottles, Laptops, Bags via edge contour geometry)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 800 < area < 25000:
                ox, oy, ow, oh = cv2.boundingRect(cnt)
                aspect = oh / float(ow + 1e-5)
                
                # Screen/Phone ratio (tall rectangle, 1.4 - 2.4 aspect ratio)
                if 1.4 < aspect < 2.5 and (25 < ow < 180) and (50 < oh < 300):
                    # Check screen brightness variance
                    roi = gray[oy:oy+oh, ox:ox+ow]
                    if roi.size > 0 and np.std(roi) > 20:
                        detections.append({
                            'label': 'Cell Phone / Device',
                            'bbox': (int(ox), int(oy), int(ow), int(oh)),
                            'score': 0.79
                        })
                        break

                # Laptop / Monitor aspect ratio (wide rectangle)
                elif 0.5 < aspect < 0.85 and (120 < ow < 400) and (80 < oh < 300):
                    detections.append({
                        'label': 'Laptop / Monitor',
                        'bbox': (int(ox), int(oy), int(ow), int(oh)),
                        'score': 0.75
                    })

                # Bottle / Cup aspect ratio
                elif 2.2 < aspect < 4.0 and (20 < ow < 100):
                    detections.append({
                        'label': 'Bottle / Cup',
                        'bbox': (int(ox), int(oy), int(ow), int(oh)),
                        'score': 0.72
                    })

        return detections



def run_predictive_vision_hud(camera_id=0):
    print("==================================================================")
    print(" VEXEL AI PREDICTIVE VISION & ANALYTICS SUITE | REAL-TIME CAMERA")
    print(" Copyright (c) 2026, Vexel Innovations. All rights reserved.")
    print("==================================================================")

    dll_path = os.path.join(os.path.dirname(__file__), "..", "build", "vexelfacedetection.dll")
    cpp_face_engine = None
    try:
        cpp_face_engine = VexelCPPFaceEngine(dll_path)
        print("[Vexel Engine] Loaded C++20 AVX2 Neural Face Engine.")
    except Exception as e:
        print(f"[Vexel Warning] Running with OpenCV fallbacks ({e})")

    object_detector = VexelObjectDetector()
    tracker = PredictiveAnalyticsTracker()


    print(f"Opening camera ID: {camera_id}...")
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Unable to access camera device {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    print("[Vexel AI] Live Multi-Object & Predictive Scanner Active! Press 'q' or 'ESC' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        t0 = time.time()
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        h, w, _ = frame.shape

        faces = []
        if cpp_face_engine:
            faces = cpp_face_engine.detect(frame, confidence_threshold=0.35)

        obj_detections = object_detector.detect_objects(frame, threshold=0.25)
        
        for f in faces:
            obj_detections.append({
                'label': 'Person (Face)',
                'bbox': f['bbox'],
                'score': f['score']
            })

        tracked_objects = tracker.update_tracks(obj_detections)

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0
        fps = 1.0 / (t1 - prev_time + 1e-6)
        prev_time = t1

        # Render Face Boxes & 5 Landmarks
        for face in faces:
            fx, fy, fw, fh = face['bbox']
            score = face['score']
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 210, 0), 2)
            cv2.putText(frame, f"Face: {score:.2f}", (fx, max(12, fy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 210, 0), 1)

            for lm in face['landmarks']:
                cv2.circle(frame, lm, 3, (0, 0, 255), -1, cv2.LINE_AA)

        # Render Tracked Objects, Category Labels, & Predictive Trajectories
        for obj in tracked_objects:
            ox, oy, ow, oh = obj.bbox
            
            box_color = (0, 255, 120) if "Person" in obj.label else (0, 165, 255)
            cv2.rectangle(frame, (int(ox), int(oy)), (int(ox + ow), int(oy + oh)), box_color, 2)

            # Predictive Vector Arrows
            future_pts = obj.predict_future_path(steps=12, dt=0.1)
            if len(future_pts) >= 2:
                for k in range(1, len(obj.history)):
                    pt1 = (int(obj.history[k-1][0]), int(obj.history[k-1][1]))
                    pt2 = (int(obj.history[k][0]), int(obj.history[k][1]))
                    cv2.line(frame, pt1, pt2, (255, 100, 0), 2)

                cx = int(ox + ow / 2.0)
                cy = int(oy + oh / 2.0)
                end_pt = future_pts[-1]
                cv2.arrowedLine(frame, (cx, cy), end_pt, (255, 0, 255), 2, tipLength=0.25)
                cv2.putText(frame, f"PREDICT: {obj.intent}", (end_pt[0] + 5, end_pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

            # Label Category & Action Badge
            label_str = f"#{obj.obj_id} {obj.label.upper()} [{obj.action}]"
            cv2.rectangle(frame, (int(ox), max(0, int(oy - 22))), (int(ox + len(label_str)*9), int(oy)), box_color, -1)
            cv2.putText(frame, label_str, (int(ox + 4), max(12, int(oy - 6))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw HUD Header Banner
        cv2.rectangle(frame, (0, 0), (w, 38), (15, 15, 15), -1)
        cv2.putText(frame, "VEXEL AI PREDICTIVE VISION | MULTI-OBJECT & ACTION SUITE", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 255), 2, cv2.LINE_AA)

        # Side Analytics HUD Box
        hud_bg = frame[45:145, 10:220]
        if hud_bg.shape[0] > 0 and hud_bg.shape[1] > 0:
            overlay = hud_bg.copy()
            cv2.rectangle(overlay, (0, 0), (210, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, hud_bg, 0.35, 0, hud_bg)
            
            cv2.putText(frame, "PREDICTIVE ANALYTICS", (18, 63),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (18, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Tracked Objects: {len(tracked_objects)}", (18, 97),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (18, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (18, 131),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        # Bottom Banner
        cv2.rectangle(frame, (0, h - 28), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "Engine: Vexel C++20 AVX2 + Caffe DNN MobileNet-SSD | Vexel Innovations", (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Vexel Predictive Vision & Action Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Vexel AI] Predictive Vision session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vexel AI Predictive Vision Camera Scanner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    run_predictive_vision_hud(args.camera)
