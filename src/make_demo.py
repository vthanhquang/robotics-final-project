"""
One-command demo preparation: regenerate every artifact used in the final
demonstration, in dependency order, with timing and a pass/fail summary.

Use this before the demo (and for a clean-clone dry-run) so nothing has to be
rendered live except the fast planner_demo.py.

Steps (detection auto-skips if the source .MOV is absent):
  1. cr_gps_drive.py        -> outputs/cr_gps_drive.mp4      (map panel)
  2. stats_panel.py         -> outputs/stats_panel.mp4       (stats panel)
  3. label_obstacles.py     -> labeled video + detection_counts.csv   [if .MOV]
  4. segment_traffic.py     -> outputs/traffic_segments.*     (real or --selftest)
  5. compose_panels.py      -> outputs/replay_3panel.mp4
  6. planner_demo.py        -> outputs/planner_compare.mp4 + planner_metrics.md
  7. evaluate_planners.py   -> outputs/planner_evaluation.{md,csv}
  8. make_path_video.py     -> path_viz/path_video.mp4        (optional extra)

Run:  python src/make_demo.py            (skip slow extras with --fast)
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PY = sys.executable
FAST = "--fast" in sys.argv

HAS_MOV = any(ROOT.glob("data/raw/*.MOV"))
COUNTS = ROOT / "data" / "outputs" / "detection_counts.csv"


def step(name, argv, outputs, required=True, skip=False, skip_reason=""):
    if skip:
        return {"name": name, "status": "skip", "secs": 0.0,
                "outputs": outputs, "reason": skip_reason, "required": required}
    print(f"\n=== {name} ===")
    t0 = time.time()
    r = subprocess.run([PY, *argv], cwd=str(ROOT),
                       capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    if not ok:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
    else:
        # echo last meaningful line
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
        if tail:
            print(tail[-1])
    return {"name": name, "status": "ok" if ok else "FAIL", "secs": dt,
            "outputs": outputs, "required": required}


def main():
    results = []
    results.append(step("map panel", ["src/cr_gps_drive.py"],
                        ["outputs/cr_gps_drive.mp4"]))
    results.append(step("stats panel", ["src/stats_panel.py"],
                        ["outputs/stats_panel.mp4"]))

    results.append(step("detection + counts", ["src/label_obstacles.py"],
                        ["data/outputs/Car_OCP_labeled.mp4",
                         "data/outputs/detection_counts.csv"],
                        required=False, skip=not HAS_MOV,
                        skip_reason="no data/raw/*.MOV on this machine"))

    seg_argv = ["src/segment_traffic.py"]
    if not COUNTS.exists():
        seg_argv.append("--selftest")
    results.append(step("traffic segmentation", seg_argv,
                        ["outputs/traffic_segments.md",
                         "outputs/traffic_segments.json"]))

    results.append(step("3-panel replay", ["src/compose_panels.py"],
                        ["outputs/replay_3panel.mp4"]))
    results.append(step("planner compare (live centerpiece)",
                        ["src/planner_demo.py"],
                        ["outputs/planner_compare.mp4",
                         "outputs/planner_metrics.md"]))
    results.append(step("planner evaluation (24 scenarios)",
                        ["src/evaluate_planners.py"],
                        ["outputs/planner_evaluation.md",
                         "outputs/planner_evaluation.csv"]))
    results.append(step("path video (extra)", ["path_viz/make_path_video.py"],
                        ["path_viz/path_video.mp4"], required=False,
                        skip=FAST, skip_reason="--fast"))
    results.append(step("presentation slides", ["src/make_slides.py"],
                        ["outputs/slide_3_results.png"]))
    results.append(step("demo reel (fallback)", ["src/demo_reel.py"],
                        ["outputs/demo_reel.mp4"]))

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"{'STEP':<34}{'STATUS':<8}{'TIME':>7}  ARTIFACT")
    print("-" * 64)
    fails = 0
    for r in results:
        mark = {"ok": "OK", "FAIL": "FAIL", "skip": "skip"}[r["status"]]
        if r["status"] == "FAIL" and r["required"]:
            fails += 1
        out0 = r["outputs"][0]
        size = ""
        p = ROOT / out0
        if r["status"] == "ok" and p.exists():
            size = f"  ({p.stat().st_size/1e6:.1f} MB)"
        line = f"{r['name']:<34}{mark:<8}{r['secs']:>6.1f}s  {out0}{size}"
        print(line)
        if r["status"] == "skip":
            print(f"{'':<34}        ({r.get('reason','')})")
    print("=" * 64)

    if not HAS_MOV:
        print("NOTE: detection skipped -> camera panel uses labeled stills, "
              "segmentation used --selftest.\n      Run label_obstacles.py on "
              "the machine with the .MOV for the real camera panel + counts.")
    if fails:
        print(f"\n{fails} REQUIRED step(s) failed -- fix before the demo.")
        sys.exit(1)
    print("\nAll required artifacts ready. For the live demo, the fast "
          "centerpiece is:\n  python src/planner_demo.py   (~2-3 s)")


if __name__ == "__main__":
    main()
