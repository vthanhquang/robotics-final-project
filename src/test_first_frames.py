"""Quick smoke test: process only the first ~5 seconds and save a preview frame."""
import cv2
from ultralytics import YOLO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_obstacles import (
    KNOWN_HEIGHTS, DEFAULT_HEIGHT, OBSTACLE_CLASSES, PALETTE,
    CONFIDENCE_THRESHOLD, estimate_focal_length, draw_label,
)

ROOT              = Path(__file__).resolve().parent.parent
INPUT             = ROOT / "data" / "raw"     / "Car_OCP_3p_2605.MOV"
PREVIEW_OUT       = ROOT / "data" / "outputs" / "preview_labeled.mp4"
PREVIEW_FRAME_OUT = ROOT / "data" / "outputs" / "preview_frame.jpg"
MODEL             = ROOT / "models" / "yolov8m.pt"
SECONDS = 5

model = YOLO(str(MODEL))
cap = cv2.VideoCapture(str(INPUT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
focal = estimate_focal_length(h)
print(f"Video {w}x{h} @ {fps:.1f}fps  focal≈{focal:.0f}px")

n_frames = int(fps * SECONDS)
out = cv2.VideoWriter(str(PREVIEW_OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

mid_frame_saved = False
for i in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break
    res = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detected = []
    for box in res.boxes:
        cls = model.names[int(box.cls[0])]
        if cls not in OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bh = y2 - y1
        if bh <= 0:
            continue
        rh = KNOWN_HEIGHTS.get(cls, DEFAULT_HEIGHT)
        dist = (rh * focal) / bh
        conf = float(box.conf[0])
        color = PALETTE[int(box.cls[0]) % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, f"{cls} {dist:.1f}m ({conf:.0%})", x1, y1, color)
        detected.append((cls, dist, conf))
    out.write(frame)
    if i == n_frames // 2 and not mid_frame_saved:
        cv2.imwrite(str(PREVIEW_FRAME_OUT), frame)
        mid_frame_saved = True
        print(f"Frame {i} detections: {detected}")

cap.release()
out.release()
print(f"Preview saved: {PREVIEW_OUT}, {PREVIEW_FRAME_OUT}")
