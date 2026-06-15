"""
Stitch the demo artifacts into ONE normalized video for the final presentation
(a reliable fallback you can pause/narrate over if a live run hiccups).

Sequence (all normalized to 1280x720 @ 30 fps):
  title card -> [1] 3-panel replay -> [2] planner baseline vs moto-aware
             -> [3] evaluation metrics card -> closing takeaways

Inputs: outputs/replay_3panel.mp4, outputs/planner_compare.mp4,
        outputs/planner_evaluation.csv   (run src/make_demo.py first)

Run:  python src/demo_reel.py            -> outputs/demo_reel.mp4
"""

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPLAY = ROOT / "outputs" / "replay_3panel.mp4"
COMPARE = ROOT / "outputs" / "planner_compare.mp4"
EVAL_CSV = ROOT / "outputs" / "planner_evaluation.csv"
OUT = ROOT / "outputs" / "demo_reel.mp4"

W, H, FPS = 1280, 720, 30
BG = (32, 30, 28)
INK = (240, 240, 240)
SUB = (170, 170, 170)
ACC = (90, 180, 250)


def writer():
    return cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"),
                           float(FPS), (W, H))


def letterbox(frame):
    ih, iw = frame.shape[:2]
    s = min(W / iw, H / ih)
    nw, nh = int(iw * s), int(ih * s)
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), (15, 15, 15), np.uint8)
    canvas[(H - nh) // 2:(H - nh) // 2 + nh, (W - nw) // 2:(W - nw) // 2 + nw] = r
    return canvas


def text(img, s, xy, scale, color, thick=1, center=False):
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (W - tw) // 2 if center else xy[0]
    cv2.putText(img, s, (x, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def card(lines, seconds, w, accent_first=True):
    """lines: list of (text, scale, color); rendered centered vertically."""
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 6), ACC, -1)
    total = sum(int(sc * 70) + 18 for _, sc, _ in lines)
    y = (H - total) // 2 + 40
    for txt, sc, col in lines:
        text(img, txt, (0, y), sc, col, 2 if sc >= 1.0 else 1, center=True)
        y += int(sc * 70) + 18
    for _ in range(int(seconds * FPS)):
        w.write(img)


def clip(path, w, t0=0.0, t1=None, speed=1.0, caption=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(t0 * fps)
    f1 = n if t1 is None else min(n, int(t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    step = max(1, int(round(speed)))
    i = f0
    while i < f1:
        ok, fr = cap.read()
        if not ok:
            break
        if (i - f0) % step == 0:
            out = letterbox(fr)
            if caption:
                cv2.rectangle(out, (0, H - 38), (W, H), (0, 0, 0), -1)
                text(out, caption, (20, H - 12), 0.6, INK, 1)
            w.write(out)
        i += 1
    cap.release()


def aggregate_eval():
    if not EVAL_CSV.exists():
        return None
    rows = list(csv.DictReader(open(EVAL_CSV)))
    out = {}
    for v in ("baseline", "conservative", "moto_aware"):
        sub = [r for r in rows if r["variant"] == v]
        n = len(sub)
        coll = sum(int(r["collision"]) for r in sub)
        pas = sum(int(r["pass_clear"]) for r in sub)
        rt = np.mean([float(r["rt"]) for r in sub])
        out[v] = (100 * coll / n, 100 * pas / n, rt)
    return out


def metrics_card(w, seconds):
    agg = aggregate_eval()
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 6), ACC, -1)
    text(img, "Does it generalize?  24 scenarios x 3 planners", (0, 90), 0.95,
         INK, 2, center=True)
    if not agg:
        text(img, "(run src/make_demo.py first)", (0, 300), 0.7, SUB, 1, True)
        for _ in range(int(seconds * FPS)):
            w.write(img)
        return
    cols = ["Planner", "Collision rate", "Clearance pass", "Mean R_T"]
    xs = [180, 540, 820, 1080]
    y = 200
    for c, x in zip(cols, xs):
        text(img, c, (x, y), 0.7, ACC, 2)
    cv2.line(img, (160, y + 14), (1180, y + 14), (90, 90, 90), 1)
    labels = {"baseline": "baseline", "conservative": "conservative",
              "moto_aware": "moto-aware (ours)"}
    for i, v in enumerate(("baseline", "conservative", "moto_aware")):
        cr, pa, rt = agg[v]
        yy = y + 70 + i * 70
        hot = v == "moto_aware"
        col = (120, 230, 140) if hot else INK
        text(img, labels[v], (xs[0], yy), 0.7, col, 2 if hot else 1)
        text(img, f"{cr:.0f}%", (xs[1], yy), 0.7, col, 2 if hot else 1)
        text(img, f"{pa:.0f}%", (xs[2], yy), 0.7, col, 2 if hot else 1)
        text(img, f"{rt:.2f}", (xs[3], yy), 0.7, col, 2 if hot else 1)
    text(img, "Safety up (38%->12% collisions), efficiency kept (R_T = baseline "
              "in safe cases)", (0, 560), 0.6, SUB, 1, True)
    for _ in range(int(seconds * FPS)):
        w.write(img)


def main():
    if not REPLAY.exists() or not COMPARE.exists():
        raise SystemExit("Missing inputs. Run: python src/make_demo.py")
    w = writer()

    card([("Vietnam-MixedTrafficSim", 1.3, INK),
          ("Adapting CommonRoad + CARLA + SUMO for VN mixed traffic", 0.6, SUB),
          ("ELEC5050 Robotics - Group 15", 0.6, ACC),
          ("Do Minh Phung  -  Vu Thanh Quang  -  Nguyen Quoc Linh", 0.55, SUB)],
         3.5, w)

    card([("1.  Data-conditioned replay", 1.1, INK),
          ("real GPS on CommonRoad map | YOLO detection | drive stats", 0.6, SUB)],
         2.5, w)
    clip(REPLAY, w, t0=0, t1=150, speed=4,
         caption="PR2: real five-trip data replayed (map | detection | stats)")

    card([("2.  Planner method improvement", 1.1, INK),
          ("baseline  vs  motorcycle-aware cost function", 0.6, SUB),
          ("motorcycle cut-in on the real VinUni road", 0.55, ACC)], 2.5, w)
    clip(COMPARE, w, speed=1,
         caption="left: baseline collides   |   right: moto-aware keeps clearance, same time")
    clip(COMPARE, w, speed=1, caption="(replay)")

    metrics_card(w, 6.0)

    card([("Takeaways", 1.1, INK),
          ("Collisions 38% -> 12% across 24 scenarios", 0.6, INK),
          ("Targeted: same efficiency as baseline when safe", 0.6, INK),
          ("Safer-and-faster than just being cautious", 0.6, INK),
          ("Next: calibrated VN-rider data (Wk 15-16)", 0.55, SUB)], 5.0, w)

    w.release()
    cap = cv2.VideoCapture(str(OUT))
    dur = cap.get(7) / max(cap.get(5), 1); cap.release()
    print(f"Done: {OUT}  ({W}x{H} @ {FPS}fps, {dur:.0f}s)")


if __name__ == "__main__":
    main()
