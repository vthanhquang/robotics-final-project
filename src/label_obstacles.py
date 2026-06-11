"""
Obstacle detection + distance estimation for first-person driving video.
Uses YOLOv8 for detection and per-class known heights for monocular distance estimation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

# ── Config ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
INPUT_VIDEO  = ROOT / "data" / "raw"     / "Car_OCP_3p_2605.MOV"
OUTPUT_VIDEO = ROOT / "data" / "outputs" / "Car_OCP_labeled.mp4"
MODEL_NAME   = ROOT / "models" / "yolov8l.pt"   # large model — better small-object recall

# Known real-world heights (metres) for common road objects
# Used for: distance_m = (real_height_m * focal_length_px) / bbox_height_px
KNOWN_HEIGHTS = {
    "person":       1.70,
    "bicycle":      1.10,
    "car":          1.50,
    "motorcycle":   1.10,
    "bus":          3.20,
    "truck":        3.50,
    "traffic light":2.50,
    "stop sign":    2.20,
    "dog":          0.50,
}
DEFAULT_HEIGHT = 1.50   # fallback for unknown classes

# Only label classes that are actually road obstacles
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "fire hydrant", "dog", "cat",
}

# Colours per class (BGR)
PALETTE = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 64, 255),
]

CONFIDENCE_THRESHOLD = 0.30
# ─────────────────────────────────────────────────────────────────────────────


def estimate_focal_length(frame_h: int) -> float:
    """
    Approximate focal length in pixels.
    iPhone wide-angle ≈ 26 mm equiv, sensor diagonal ≈ 12 mm, 1080p crop.
    Empirical rule: focal_px ≈ frame_height * 1.2 for iPhone footage.
    """
    return frame_h * 1.2


def draw_label(img, text, x1, y1, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    bg_y1 = max(y1 - th - baseline - 4, 0)
    cv2.rectangle(img, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - baseline - 1),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def process_video(input_path: str, output_path: str):
    model = YOLO(str(MODEL_NAME))
    cap   = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        sys.exit(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    focal  = estimate_focal_length(height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Video: {width}x{height} @ {fps:.1f} fps  |  {total} frames")
    print(f"Estimated focal length: {focal:.0f} px")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        for i, box in enumerate(results.boxes):
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]

            if cls_name not in OBSTACLE_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox_h = y2 - y1
            if bbox_h <= 0:
                continue

            real_h   = KNOWN_HEIGHTS.get(cls_name, DEFAULT_HEIGHT)
            distance = (real_h * focal) / bbox_h   # metres

            color = PALETTE[cls_id % len(PALETTE)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {distance:.1f}m ({conf:.0%})"
            draw_label(frame, label, x1, y1, color)

        # Progress overlay
        info = f"Frame {frame_idx+1}/{total}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx}/{total} frames …")

    cap.release()
    out.release()
    print(f"\nDone. Output saved to: {output_path}")


if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else INPUT_VIDEO
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_VIDEO
    process_video(input_path, output_path)
