"""
Split a trip into low-traffic and high-traffic segments from per-frame
detection counts, following Progress Report 2 (Sec. 2.3 / Table 2):

    mean detected actors per frame <= 4  -> low traffic
    mean detected actors per frame  > 4  -> high traffic

Input:  data/outputs/detection_counts.csv   (written by src/label_obstacles.py)
        columns: frame,t,n_total,n_car,n_motorcycle,n_person,nearest_m

Output: outputs/traffic_segments.md     human-readable segment table
        outputs/traffic_segments.json   segment list (start/end/mode) for the
                                        selectable demo modes + compose_panels.py
        outputs/traffic_segments.png    colored timeline (low=green, high=red)

Run:  python src/segment_traffic.py              (needs the counts CSV)
      python src/segment_traffic.py --selftest   (synthetic counts, validates logic)
"""

import csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COUNTS = ROOT / "data" / "outputs" / "detection_counts.csv"
OUT_MD = ROOT / "outputs" / "traffic_segments.md"
OUT_JSON = ROOT / "outputs" / "traffic_segments.json"
OUT_PNG = ROOT / "outputs" / "traffic_segments.png"

WINDOW_S   = 5.0      # segment granularity
THRESHOLD  = 4.0      # PR2 Table 2: mean actors/frame > 4 => high traffic


def load_counts(path):
    t, n, cls = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"]))
            n.append(float(r["n_total"]))
            cls.append((float(r.get("n_car", 0)), float(r.get("n_motorcycle", 0)),
                        float(r.get("n_person", 0))))
    return np.array(t), np.array(n), np.array(cls)


def synth_counts():
    """Synthetic per-frame counts for --selftest (clearly NOT real data)."""
    fps, dur = 30, 219.9
    t = np.arange(0, dur, 1 / fps)
    rng = np.random.default_rng(0)
    base = (2.5 + 4.0 * (np.sin(t / 18) > 0.3)          # bursts of dense flow
            + 2.0 * (np.sin(t / 7 + 1) > 0.6))
    n = np.clip(base + rng.normal(0, 0.8, t.size), 0, None)
    car = n * 0.45; moto = n * 0.45; ped = n * 0.10
    return t, n, np.stack([car, moto, ped], 1)


def segment(t, n, cls):
    dur = t[-1]
    n_win = int(np.ceil(dur / WINDOW_S))
    rows = []
    for w in range(n_win):
        a, b = w * WINDOW_S, min((w + 1) * WINDOW_S, dur)
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        mean_n = float(n[m].mean())
        mode = "high" if mean_n > THRESHOLD else "low"
        cmix = cls[m].mean(0)
        dom = ["car", "motorcycle", "pedestrian"][int(np.argmax(cmix))]
        rows.append({"t_start": a, "t_end": b, "mean_actors": mean_n,
                     "mode": mode, "dominant": dom})
    # merge consecutive windows of same mode
    merged = []
    for r in rows:
        if merged and merged[-1]["mode"] == r["mode"]:
            p = merged[-1]
            span_p = p["t_end"] - p["t_start"]
            span_r = r["t_end"] - r["t_start"]
            p["mean_actors"] = (p["mean_actors"] * span_p +
                                r["mean_actors"] * span_r) / (span_p + span_r)
            p["t_end"] = r["t_end"]
        else:
            merged.append(dict(r))
    return merged


def write_outputs(segs, note=""):
    OUT_JSON.write_text(json.dumps(
        [{"t_start": round(s["t_start"], 1), "t_end": round(s["t_end"], 1),
          "mode": s["mode"]} for s in segs], indent=2))

    low = [s for s in segs if s["mode"] == "low"]
    high = [s for s in segs if s["mode"] == "high"]
    low_t = sum(s["t_end"] - s["t_start"] for s in low)
    high_t = sum(s["t_end"] - s["t_start"] for s in high)

    lines = [f"# Traffic segmentation — low vs high{note}", "",
             f"Rule (PR2 Table 2): mean detected actors/frame > {THRESHOLD:.0f} "
             f"= high traffic, otherwise low. Window = {WINDOW_S:.0f} s.", "",
             f"- **Low-traffic total:** {low_t:.0f} s in {len(low)} segments",
             f"- **High-traffic total:** {high_t:.0f} s in {len(high)} segments",
             "", "| # | start | end | dur | mean actors | dominant | mode |",
             "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(segs, 1):
        lines.append(f"| {i} | {s['t_start']:.0f}s | {s['t_end']:.0f}s | "
                     f"{s['t_end']-s['t_start']:.0f}s | {s['mean_actors']:.1f} | "
                     f"{s['dominant']} | **{s['mode']}** |")
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))

    _timeline_png(segs)
    print(f"\nWrote {OUT_MD.name}, {OUT_JSON.name}, {OUT_PNG.name}")


def _timeline_png(segs):
    import cv2
    W, H = 1000, 90
    dur = segs[-1]["t_end"]
    img = np.full((H, W, 3), 255, np.uint8)
    for s in segs:
        x0 = int(s["t_start"] / dur * W)
        x1 = int(s["t_end"] / dur * W)
        col = (60, 170, 60) if s["mode"] == "low" else (50, 50, 210)
        cv2.rectangle(img, (x0, 20), (x1, 60), col, -1)
        cv2.rectangle(img, (x0, 20), (x1, 60), (255, 255, 255), 1)
    cv2.putText(img, "low", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 170, 60), 2)
    cv2.putText(img, "high", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 210), 2)
    cv2.putText(img, "0s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.putText(img, f"{dur:.0f}s", (W - 40, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (80, 80, 80), 1)
    cv2.imwrite(str(OUT_PNG), img)


def main():
    selftest = "--selftest" in sys.argv
    if selftest:
        print("SELFTEST: synthetic counts (not real detection data)\n")
        t, n, cls = synth_counts()
        note = "  *(SELFTEST — synthetic counts)*"
    elif COUNTS.exists():
        t, n, cls = load_counts(COUNTS)
        note = ""
    else:
        sys.exit(f"No {COUNTS.relative_to(ROOT)} found.\n"
                 f"Run detection first:  python src/label_obstacles.py\n"
                 f"or validate the tool:  python src/segment_traffic.py --selftest")
    segs = segment(t, n, cls)
    write_outputs(segs, note)


if __name__ == "__main__":
    main()
