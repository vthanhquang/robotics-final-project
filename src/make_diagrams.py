"""
Draw two slide figures (light theme, VinUni red) with matplotlib:
  outputs/arch_pipeline.png  - end-to-end system architecture
  outputs/apex_model.png     - APEX planner internal model flow

Run:  python src/make_diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

RED = "#b21f1f"
INK = "#222222"
GREEN = "#1e7a34"
BLUE = "#1f5fb2"
GREY = "#666666"


def box(ax, x, y, w, h, title, lines, fill, title_color=RED, hot=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2.4 if hot else 1.4,
                                edgecolor=GREEN if hot else "#cccccc",
                                facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="top",
            fontsize=13.5, fontweight="bold", color=title_color, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.78 - i * 0.42, ln, ha="left", va="top",
                fontsize=10.2, color=INK, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=2.0,
                                 color=color, zorder=1))


def fig(w, h):
    f, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, 16); ax.set_ylim(0, h); ax.axis("off")
    f.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return f, ax


# ── 1. System architecture ────────────────────────────────────────────────────
def architecture():
    f, ax = fig(16, 7)
    ax.text(8, 6.7, "Vietnam-MixedTrafficSim — system architecture",
            ha="center", fontsize=17, fontweight="bold", color=RED)

    y, h = 3.4, 2.5
    cols = [
        (0.2, "1. Real data", ["5 car trips (VinUni /", "Ocean Park)",
                                "video + GPS + IMU", "(iPhone, 100 Hz)"], "#eaf2fb"),
        (3.3, "2. Perception + map", ["YOLOv8: car / moto /", "pedestrian",
                                      "OSM -> Lanelet2 /", "CommonRoad map"], "#eafaf0"),
        (6.4, "3. Behaviour model", ["moto calibration: 4", "classes (cut-in,",
                                     "lane-split, follow,", "cross) -> agents"], "#fdf3e7"),
        (9.5, "4. Scenarios", ["52 data-conditioned", "cases, 8 families",
                               "low / high traffic", "(>4 actors/frame)"], "#f2f2f2"),
        (12.6, "5. Planners", ["IDM, ORCA/VO,", "baseline, conservative,",
                               "moto-aware,", "APEX (ours)"], "#fdeaea"),
    ]
    w = 2.8
    for x, t, lines, fill in cols:
        box(ax, x, y, w, h, t, lines, fill, hot=(t.startswith("5")))
    for i in range(len(cols) - 1):
        arrow(ax, cols[i][0] + w, y + h / 2, cols[i + 1][0], y + h / 2)

    # evaluation strip
    box(ax, 3.3, 0.5, 9.5, 1.7, "6. Evaluation (CommonRoad metrics)",
        ["collision = 0   ·   min clearance > 0.3 m   ·   TTC exposure < 2 s   ·   "
         "AEB = 0   ·   efficiency R_T <= 1.25   ·   TR1 competition cost"], "#eef3f8",
        title_color=BLUE)
    arrow(ax, 13.9, y, 12.0, 2.2)   # planners -> evaluation
    f.savefig(OUT / "arch_pipeline.png"); plt.close(f)
    print("wrote arch_pipeline.png")


# ── 2. APEX model flow ─────────────────────────────────────────────────────────
def apex_model():
    f, ax = fig(16, 6)
    ax.text(8, 5.7, "APEX — predictive risk-aware planner (model flow)",
            ha="center", fontsize=17, fontweight="bold", color=RED)

    y, h, w = 2.6, 2.3, 2.55
    xs = [0.2, 3.35, 6.5, 9.65, 12.8]
    blocks = [
        ("Inputs", ["ego state (s, d, v)", "moto observations:", "s, d, lateral vel ḋ"], "#eaf2fb", RED, False),
        ("Predict + reachability", ["const-velocity +", "envelope for adjacent", "/ crossing motos"], "#eafaf0", RED, False),
        ("Multi-horizon candidates", ["Frenet quintic/quartic", "T = 1.5 / 2.5 / 4.0 s", "(brake fast .. full speed)"], "#fdf3e7", RED, False),
        ("Safety filter", ["hard footprint-clearance", "margin (all motos, all t)", "+ junction-yield cap"], "#fdeaea", RED, False),
        ("Speed-max select", ["fastest feasible traj.", "-> output, replan 0.1 s", "collision-free by constr."], "#eafaf0", GREEN, True),
    ]
    for x, (t, lines, fill, tc, hot) in zip(xs, blocks):
        box(ax, x, y, w, h, t, lines, fill, title_color=tc, hot=hot)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    caps = [("anticipates cut-ins\n(vs blind const-position)", 4.6),
            ("recovers efficiency\n(vs IDM over-braking)", 7.75),
            ("0 collisions\n(vs point-box flaw)", 10.9),
            ("best efficiency among\nsafe planners", 14.05)]
    for txt, x in caps:
        ax.text(x, 2.1, txt, ha="center", va="top", fontsize=9, color=GREEN, style="italic")
    f.savefig(OUT / "apex_model.png"); plt.close(f)
    print("wrote apex_model.png")


if __name__ == "__main__":
    architecture()
    apex_model()
