"""
Bar charts of the 52-scenario benchmark for the slides:
  outputs/bench_bars.png        - collision rate + travel-time R_T per planner
  outputs/apex_improvement.png  - APEX % improvement vs each baseline

Numbers: src/discriminating_suite.py (52 scenarios, 6 planners).
Run:  python src/make_charts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

RED, GREEN, ORANGE, GREY, INK = "#cc3b3b", "#1e9e4a", "#e08a1e", "#9aa6b2", "#222222"

PLANNERS = ["IDM", "ORCA/VO", "Baseline", "Conserv.", "Moto-aware", "APEX"]
COLL = [5, 1, 28, 0, 13, 0]          # collisions /52
RT   = [1.57, 1.34, 1.27, 1.86, 1.33, 1.43]
N = 52
coll_pct = [c / N * 100 for c in COLL]

def colors(values, worst_is_max=True):
    cols = []
    for p in PLANNERS:
        if p == "APEX":
            cols.append(GREEN)
        elif p == "Baseline":
            cols.append(RED)
        elif p == "Conserv.":
            cols.append(ORANGE)
        else:
            cols.append(GREY)
    return cols


def bench_bars():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.3, 5.2), dpi=150)
    fig.suptitle("APEX is the only planner that is safe AND efficient (52 scenarios)",
                 fontsize=16, fontweight="bold", color="#b21f1f")

    b1 = ax1.bar(PLANNERS, coll_pct, color=colors(coll_pct))
    ax1.set_title("Collision rate  —  lower is better", fontsize=12, color=INK)
    ax1.set_ylabel("% of 52 scenarios with a collision")
    ax1.set_ylim(0, 60)
    for r, v, c in zip(b1, coll_pct, COLL):
        ax1.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f}%\n({c}/52)",
                 ha="center", va="bottom", fontsize=9, color=INK)

    b2 = ax2.bar(PLANNERS, RT, color=colors(RT))
    ax2.set_title("Travel-time ratio R_T  —  lower is faster", fontsize=12, color=INK)
    ax2.set_ylabel("R_T = AV time / human time")
    ax2.set_ylim(0, 2.1)
    ax2.axhline(1.0, color="#bbbbbb", lw=1, ls="--")
    for r, v in zip(b2, RT):
        ax2.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9, color=INK)
    # APEX vs conservative annotation (both crash-free)
    ax2.annotate("23% faster than\nconservative\n(same 0 collisions)",
                 xy=(5, 1.43), xytext=(3.4, 0.55), fontsize=9, color=GREEN,
                 ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "bench_bars.png"); plt.close(fig)
    print("wrote bench_bars.png")


def apex_improvement():
    labels = ["Collisions vs\nnaive baseline (28→0)",
              "Collisions vs our\nprevious moto-aware (13→0)",
              "Clearance-fails\nvs baseline (30→0)",
              "Faster than conservative\n(same 0 collisions)",
              "Faster than IDM\n(and 0 vs 5 collisions)"]
    vals = [100, 100, 100, 23, 9]
    fig, ax = plt.subplots(figsize=(13.3, 5.0), dpi=150)
    fig.suptitle("APEX improvement (%)", fontsize=16, fontweight="bold", color="#b21f1f")
    bars = ax.barh(labels[::-1], vals[::-1], color=GREEN)
    ax.set_xlim(0, 112); ax.set_xlabel("% improvement (higher is better)")
    for r, v in zip(bars, vals[::-1]):
        ax.text(v + 1.5, r.get_y() + r.get_height() / 2,
                f"{v:.0f}%" + ("" if v == 100 else " faster"),
                va="center", fontsize=11, fontweight="bold", color=GREEN)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "apex_improvement.png"); plt.close(fig)
    print("wrote apex_improvement.png")


if __name__ == "__main__":
    bench_bars()
    apex_improvement()
