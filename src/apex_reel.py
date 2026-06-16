"""
Stitch the six APEX scenario-comparison videos (Quang's planner-scenario-suite
run) into one narratable demo reel for the presentation.

Each source video is a 4-panel comparison
(baseline Frenet | ORCA/VO | moto-aware | APEX) where the others collide and
APEX stays safe. This reel wraps them with a title card, a results / %-improvement
card, and a closing card. Source frames are held to play back in real time
(the source videos are 10 fps; the reel is 30 fps).

Inputs (outputs/, from running Quang's src/scenario_videos.py):
  quang_cut_in_C07, quang_crossing_X01, quang_shoulder_merge_H01,
  quang_multi_lane_W01, quang_two_moto_T2-05, quang_three_moto_T3-03

Run:  python src/apex_reel.py   ->  outputs/apex_reel.mp4
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "apex_reel.mp4"

W, H, FPS = 1280, 720, 30
BG = (32, 30, 28)
INK = (240, 240, 240)
SUB = (170, 170, 170)
ACC = (90, 180, 250)
GREEN = (120, 230, 140)
F = cv2.FONT_HERSHEY_SIMPLEX

VIDEOS = [
    ("Phung_cut_in_C07",         "Cut-in"),
    ("Phung_crossing_X01",       "Junction crossing"),
    ("Phung_shoulder_merge_H01", "Shoulder merge"),
    ("Phung_multi_lane_W01",     "Multi-lane weave"),
    ("Phung_two_moto_T2-05",     "Two motorcycles"),
    ("Phung_three_moto_T3-03",   "Three motorcycles"),
]


def _text(img, s, y, scale, color, thick=1):
    (tw, _), _ = cv2.getTextSize(s, F, scale, thick)
    cv2.putText(img, s, ((W - tw) // 2, y), F, scale, color, thick, cv2.LINE_AA)


def card(lines, seconds, w):
    """lines: list of (text, scale, color), vertically centered."""
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 6), ACC, -1)
    total = sum(int(sc * 70) + 18 for _, sc, _ in lines)
    y = (H - total) // 2 + 40
    for txt, sc, col in lines:
        _text(img, txt, y, sc, col, 2 if sc >= 1.0 else 1)
        y += int(sc * 70) + 18
    for _ in range(int(seconds * FPS)):
        w.write(img)


def letterbox(frame):
    ih, iw = frame.shape[:2]
    s = min(W / iw, H / ih)
    nw, nh = int(iw * s), int(ih * s)
    r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), (15, 15, 15), np.uint8)
    canvas[(H - nh) // 2:(H - nh) // 2 + nh, (W - nw) // 2:(W - nw) // 2 + nw] = r
    return canvas


def play(path, caption, w):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10
    rep = max(1, round(FPS / src_fps))          # hold each frame -> real time
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out = letterbox(fr)
        cv2.rectangle(out, (0, H - 40), (W, H), (0, 0, 0), -1)
        cv2.putText(out, caption, (24, H - 13), F, 0.6, INK, 1, cv2.LINE_AA)
        for _ in range(rep):
            w.write(out)
    cap.release()


def main():
    w = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"),
                        float(FPS), (W, H))

    card([("APEX  -  predictive risk-aware planner", 1.2, INK),
          ("vs baseline Frenet | ORCA/VO | moto-aware,  on real VinUni roads", 0.55, SUB),
          ("ELEC5050 Robotics - Group 15", 0.55, ACC)], 3.5, w)

    card([("52 scenarios x 6 planners", 1.0, INK),
          ("0 collisions / 0 clearance-fails", 0.7, GREEN),
          ("-100% vs baseline Frenet (28) and vs our previous moto-aware (13)", 0.55, INK),
          ("23% faster than the only other crash-free planner (R_T 1.43 vs 1.86)", 0.55, INK),
          ("beats published IDM & ORCA baselines; crossings solved 0/4", 0.55, INK)],
         6.0, w)

    card([("Live comparison", 1.1, INK),
          ("each panel:  baseline | ORCA/VO | moto-aware | APEX", 0.55, SUB),
          ("others collide  ->  APEX stays safe", 0.6, ACC)], 3.0, w)

    n = 0
    for name, label in VIDEOS:
        p = ROOT / "outputs" / f"{name}.mp4"
        if not p.exists():
            print(f"  (skip, missing) {name}")
            continue
        play(p, f"{label}  -  others collide, APEX stays safe", w)
        n += 1

    card([("Takeaways", 1.1, INK),
          ("APEX is the only safe AND efficient planner", 0.6, GREEN),
          ("predict + reachability + hard clearance + junction-yield", 0.55, INK),
          ("next: calibrate from 200 h moto + 5 car-trip data", 0.55, SUB)], 5.0, w)

    w.release()
    cap = cv2.VideoCapture(str(OUT))
    dur = cap.get(7) / max(cap.get(5), 1); cap.release()
    print(f"Done: {OUT}  ({W}x{H} @ {FPS}fps, {dur:.0f}s, {n}/6 videos)")


if __name__ == "__main__":
    main()
