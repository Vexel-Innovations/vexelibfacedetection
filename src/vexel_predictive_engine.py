"""
Vexel Predictive Vision & Motion Analytics Engine
Copyright (c) 2026, Vexel Innovations. All rights reserved.

Provides real-time trajectory extrapolation, velocity vectors, and behavioral intent analysis.
"""

import time
import math
import numpy as np

class TrackedObject:
    def __init__(self, obj_id, label, bbox, confidence):
        self.obj_id = obj_id
        self.label = label
        self.bbox = bbox  # (x, y, w, h)
        self.confidence = confidence
        
        # Center coordinates
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.history = [(cx, cy, time.time())]
        
        # Motion vectors (dx/dt, dy/dt)
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.intent = "Stationary"
        self.action = "Standing"

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0

        now = time.time()
        self.history.append((cx, cy, now))
        if len(self.history) > 30:
            self.history.pop(0)

        # Calculate velocity over recent points
        if len(self.history) >= 2:
            prev_cx, prev_cy, prev_t = self.history[-2]
            dt = max(now - prev_t, 1e-4)
            dx = cx - prev_cx
            dy = cy - prev_cy

            # Smoothing velocity
            alpha = 0.6
            self.vx = alpha * (dx / dt) + (1 - alpha) * self.vx
            self.vy = alpha * (dy / dt) + (1 - alpha) * self.vy
            self.speed = math.sqrt(self.vx**2 + self.vy**2)

            # Analyze intent
            if self.speed < 15.0:
                self.intent = "Stationary / Loitering"
                self.action = "Sitting / Idle" if self.bbox[3] < self.bbox[2] * 1.2 else "Standing"
            else:
                if dy > 5.0 and abs(dx) < abs(dy) * 1.5:
                    self.intent = "Approaching Camera"
                    self.action = "Walking Forward"
                elif dy < -5.0:
                    self.intent = "Moving Away"
                    self.action = "Walking Away"
                elif dx > 10.0:
                    self.intent = "Moving Right"
                    self.action = "Traversing"
                elif dx < -10.0:
                    self.intent = "Moving Left"
                    self.action = "Traversing"
                else:
                    self.intent = "Active Movement"
                    self.action = "In Motion"

    def predict_future_path(self, steps=10, dt=0.1):
        """Predict future (x, y) coordinates for next N steps."""
        if len(self.history) < 2 or self.speed < 5.0:
            return []

        cx = self.bbox[0] + self.bbox[2] / 2.0
        cy = self.bbox[1] + self.bbox[3] / 2.0

        future_pts = []
        for i in range(1, steps + 1):
            pred_x = cx + self.vx * (i * dt)
            pred_y = cy + self.vy * (i * dt)
            future_pts.append((int(pred_x), int(pred_y)))

        return future_pts


class PredictiveAnalyticsTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 1

    def update_tracks(self, detections):
        """
        Detections: list of dicts {'label': str, 'bbox': (x,y,w,h), 'score': float}
        """
        updated_ids = set()

        for det in detections:
            bbox = det['bbox']
            label = det['label']
            score = det['score']
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0

            # Find closest existing track
            best_id = None
            best_dist = 80.0  # Max pixel distance for association

            for obj_id, obj in self.tracked_objects.items():
                if obj_id in updated_ids or obj.label != label:
                    continue
                last_cx, last_cy, _ = obj.history[-1]
                dist = math.hypot(cx - last_cx, cy - last_cy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id

            if best_id is not None:
                self.tracked_objects[best_id].update(bbox, score)
                updated_ids.add(best_id)
            else:
                new_obj = TrackedObject(self.next_id, label, bbox, score)
                self.tracked_objects[self.next_id] = new_obj
                updated_ids.add(self.next_id)
                self.next_id += 1

        # Clean up lost tracks
        now = time.time()
        lost_ids = [obj_id for obj_id, obj in self.tracked_objects.items()
                    if obj_id not in updated_ids and (now - obj.history[-1][2]) > 2.0]
        for obj_id in lost_ids:
            del self.tracked_objects[obj_id]

        return list(self.tracked_objects.values())
