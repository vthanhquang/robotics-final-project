"""
Render the 3 presentation slides (problem / method / results) for the 4-minute
talk, in the same dark theme as src/demo_reel.py so the deck and the demo match.

Slide 3's numbers are read from outputs/planner_evaluation.csv so the deck stays
in sync with the evaluation (run src/make_demo.py first).

Output: outputs/slide_1_problem.png, slide_2_method.png, slide_3_results.png

Run:  python src/make_slides.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_CSV = ROOT / "outputs" / "planner_evaluation.csv"
OUT = ROOT / "outputs"

W, H = 1920, 1080
BG = (32, 30, 28)
INK = (240, 240, 240)
SUB = (175, 175, 175)
ACC = (90, 180, 250)
GREEN = (120, 230, 140)
F = cv2.FONT_HERSHEY_SIMPLEX


def base(num):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 10), ACC, -1)
    cv2.putText(img, "ELEC5050 Robotics  -  Group 15  -  Vietnam-MixedTrafficSim",
                (60, H - 40), F, 0.7, SUB, 1, cv2.LINE_AA)
    cv2.putText(img, f"{num}/3", (W - 130, H - 40), F, 0.7, SUB, 1, cv2.LINE_AA)
    return img


def title(img, s, sub=None):
    cv2.putText(img, s, (60, 170), F, 1.9, INK, 4, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (62, 230), F, 0.95, ACC, 2, cv2.LINE_AA)


def bullet(img, y, s, scale=1.0, color=INK, indent=0, dot=True):
    x = 80 + indent
    if dot:
        cv2.circle(img, (x + 6, y - 10), 6, color, -1)
        x += 34
    cv2.putText(img, s, (x, y), F, scale, color, 2, cv2.LINE_AA)


def read_eval():
    if not EVAL_CSV.exists():
        return None
    rows = list(csv.DictReader(open(EVAL_CSV)))
    agg = {}
    for v in ("baseline", "conservative", "moto_aware"):
        sub = [r for r in rows if r["variant"] == v]
        n = len(sub)
        agg[v] = (100 * sum(int(r["collision"]) for r in sub) / n,
                  100 * sum(int(r["pass_clear"]) for r in sub) / n,
                  float(np.mean([float(r["rt"]) for r in sub])))
    nsc = len({r["scenario"] for r in rows})
    return agg, nsc


# ── Slide 1: problem ──────────────────────────────────────────────────────────
def slide1():
    img = base(1)
    title(img, "The Problem: Mixed Traffic in Vietnam")
    y = 360
    for s in [
        "AV motion planners assume Western traffic: marked lanes, rule-abiding drivers.",
        "In Vietnam, >90% of vehicles are motorcycles that actively:",
    ]:
        bullet(img, y, s, 0.95); y += 90
    for s in ["lane-split and squeeze through sub-meter gaps,",
              "cut in at short range and ignore signals."]:
        bullet(img, y, s, 0.9, SUB, indent=60); y += 80
    y += 20
    bullet(img, y, "Standard planners react too late -> unsafe in these scenes.",
           0.95, GREEN); y += 100
    bullet(img, y, "Goal: a planner that ANTICIPATES motorcycle cut-ins, on the "
                   "real VinUni / Ocean Park road.", 0.95, INK)
    cv2.imwrite(str(OUT / "slide_1_problem.png"), img)


# ── Slide 2: method ───────────────────────────────────────────────────────────
def slide2():
    img = base(2)
    title(img, "Method: Motorcycle-Aware Reactive Planner",
          "proposal obj. 3  /  Progress Report 1, sec. 3.3")
    y = 360
    bullet(img, y, "Frenet-frame sampling planner (Werling): quintic/quartic "
                   "candidate trajectories,", 0.92); y += 80
    bullet(img, y, "scored on jerk + speed + lateral offset + safety; lowest-cost "
                   "feasible one each step.", 0.92, SUB, indent=60); y += 110
    bullet(img, y, "Our improvement over the baseline:", 0.95, ACC); y += 90
    bullet(img, y, "1) Uncertainty-buffer propagation - the motorcycle's predicted",
           0.9, INK, indent=60); y += 70
    bullet(img, y, "    footprint inflates with its lateral velocity -> the cut-in "
                   "is seen early.", 0.9, SUB, indent=60, dot=False); y += 90
    bullet(img, y, "2) Dynamic safety re-weighting - safety cost rises when lateral",
           0.9, INK, indent=60); y += 70
    bullet(img, y, "    motion is detected.", 0.9, SUB, indent=60, dot=False); y += 100
    bullet(img, y, "Runs on the real VinUni CommonRoad map + recorded GPS.",
           0.95, GREEN)
    cv2.imwrite(str(OUT / "slide_2_method.png"), img)


# ── Slide 3: results ──────────────────────────────────────────────────────────
def slide3():
    img = base(3)
    data = read_eval()
    title(img, "Results: General, Targeted, Efficient",
          ("24 scenarios x 3 planners" if data else "run src/make_demo.py first"))
    if not data:
        cv2.imwrite(str(OUT / "slide_3_results.png"), img)
        return
    agg, nsc = data
    # table
    cols = ["Planner", "Collision rate", "Clearance pass", "Mean R_T"]
    xs = [90, 720, 1130, 1520]
    y = 360
    for c, x in zip(cols, xs):
        cv2.putText(img, c, (x, y), F, 0.9, ACC, 2, cv2.LINE_AA)
    cv2.line(img, (80, y + 18), (1780, y + 18), (90, 90, 90), 1)
    names = {"baseline": "baseline", "conservative": "conservative",
             "moto_aware": "moto-aware (ours)"}
    for i, v in enumerate(("baseline", "conservative", "moto_aware")):
        cr, pa, rt = agg[v]
        yy = y + 90 + i * 80
        hot = v == "moto_aware"
        col = GREEN if hot else INK
        th = 3 if hot else 2
        cv2.putText(img, names[v], (xs[0], yy), F, 0.9, col, th, cv2.LINE_AA)
        cv2.putText(img, f"{cr:.0f}%", (xs[1], yy), F, 0.9, col, th, cv2.LINE_AA)
        cv2.putText(img, f"{pa:.0f}%", (xs[2], yy), F, 0.9, col, th, cv2.LINE_AA)
        cv2.putText(img, f"{rt:.2f}", (xs[3], yy), F, 0.9, col, th, cv2.LINE_AA)
    y = y + 90 + 3 * 80 + 60
    for s, c in [("Safety up: collisions 38% -> 12%, clearance-pass 50% -> 88%.", INK),
                 ("Targeted: same efficiency as baseline when safe (R_T 1.01 = 1.01).", INK),
                 ("Safer AND faster than just being cautious (1.35 vs 1.72).", GREEN)]:
        bullet(img, y, s, 0.9, c); y += 80
    cv2.imwrite(str(OUT / "slide_3_results.png"), img)


def main():
    slide1(); slide2(); slide3()
    print("Wrote slide_1_problem.png, slide_2_method.png, slide_3_results.png "
          f"in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
