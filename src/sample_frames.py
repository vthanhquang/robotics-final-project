"""Extract a few sample frames from the labeled video for visual verification."""
import cv2
import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "data" / "outputs" / "Car_OCP_labeled.mp4"
OUTDIR = ROOT / "data" / "outputs"

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    sys.exit(f"Cannot open {VIDEO}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Labeled video: {total} frames @ {fps:.1f} fps  ({total/fps:.1f}s)")

for p in [0.10, 0.30, 0.50, 0.70, 0.90]:
    idx = int(total * p)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    out = OUTDIR / f"sample_{int(p*100):02d}pct.jpg"
    cv2.imwrite(str(out), frame)
    print(f"  frame {idx} -> {out.name}")

cap.release()
