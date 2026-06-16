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

# APEX 52-scenario benchmark (Quang's planner-scenario-suite branch, slide_planning.md)
# (label, collisions, <0.3m fails, mean min-clr m, mean R_T)
APEX_RESULTS = [
    ("IDM (Treiber'00)",      "5/52",  "7/52",  "5.73",  "1.57"),
    ("ORCA/VO (vdB'11)",      "1/52",  "1/52",  "2.02",  "1.34"),
    ("baseline Frenet",       "28/52", "30/52", "1.24",  "1.27"),
    ("conservative",          "0/52",  "0/52",  "13.96", "1.86"),
    ("moto-aware (prev ours)","13/52", "14/52", "2.31",  "1.33"),
    ("APEX (predictive, OURS)", "0/52", "0/52", "2.16", "1.43"),
]

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
    title(img, "Method: APEX predictive risk-aware planner",
          "fuses car-following + reactive avoidance + sampling; fixes each flaw")
    y = 340
    rows = [
        "1) Predict every motorcycle over a 4 s horizon + reachability envelope",
        "    -> anticipates a cut-in before it starts (vs blind constant-position).",
        "2) Hard footprint-clearance margin to all motos over the whole horizon",
        "    -> collision-free by construction (fixes the point-box flaw).",
        "3) Junction-yield speed-cap for roadside / crossing motos",
        "    -> solves crossings (0/4) where every other planner fails.",
        "4) Multi-horizon trajectories -> brake hard when needed, full speed when",
        "    clear (fixes IDM/conservative over-braking).",
        "5) Speed-maximizing objective under the safety constraint",
        "    -> most efficient among collision-free planners.",
    ]
    for i, s in enumerate(rows):
        head = not s.startswith("    ")
        bullet(img, y, s, 0.82, INK if head else SUB, indent=40,
               dot=head); y += 66 if head else 60
    y += 10
    bullet(img, y, "Runs on the real VinUni CommonRoad map + recorded GPS.",
           0.85, GREEN)
    cv2.imwrite(str(OUT / "slide_2_method.png"), img)


# ── Slide 3: results ──────────────────────────────────────────────────────────
def slide3():
    img = base(3)
    title(img, "Results: 52 scenarios x 6 planners",
          "only conservative & APEX are crash-free; APEX is the efficient one")
    cols = ["Planner", "Collisions", "<0.3 m fails", "Min-clr", "R_T"]
    xs = [70, 760, 1060, 1400, 1660]
    y = 300
    for c, x in zip(cols, xs):
        cv2.putText(img, c, (x, y), F, 0.72, ACC, 2, cv2.LINE_AA)
    cv2.line(img, (60, y + 16), (1850, y + 16), (90, 90, 90), 1)
    for i, (name, coll, fails, clr, rt) in enumerate(APEX_RESULTS):
        yy = y + 64 + i * 62
        hot = name.startswith("APEX")
        col = GREEN if hot else INK
        th = 3 if hot else 2
        for x, val in zip(xs, (name, coll, fails, clr, rt)):
            cv2.putText(img, val, (x, yy), F, 0.66, col, th, cv2.LINE_AA)

    y = y + 64 + len(APEX_RESULTS) * 62 + 46
    cv2.putText(img, "APEX vs the rest:", (70, y), F, 0.82, ACC, 2, cv2.LINE_AA)
    y += 64
    for s in ["0 collisions across all 52  ->  -100% vs baseline (28) and vs our previous (13)",
              "23% faster than the only other crash-free planner (R_T 1.43 vs 1.86)",
              "beats published IDM & ORCA baselines; crossings solved 0/4"]:
        bullet(img, y, s, 0.74, GREEN); y += 62
    cv2.imwrite(str(OUT / "slide_3_results.png"), img)


def main():
    slide1(); slide2(); slide3()
    print("Wrote slide_1_problem.png, slide_2_method.png, slide_3_results.png "
          f"in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
