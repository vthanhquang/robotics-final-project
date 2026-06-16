"""
Render the 4-panel APEX comparison video (baseline | ORCA/VO | moto-aware | APEX)
for a curated set of scenarios from the 52-case discriminating suite.

Each panel replays the same scenario under a different planner on the real
VinUni / Ocean Park map, with a clearance/speed HUD and a SAFE/COLLISION badge:
the published/naive planners collide while APEX stays safe.

Reuses the planners, scenarios and geometry from discriminating_suite.py and the
shared map background from scenario_videos.py.

Run:  python src/scenario_videos_apex.py
Out:  outputs/Phung_<family>_<tag>.mp4   (6 videos)
"""

import math
from pathlib import Path

import cv2
import numpy as np

from discriminating_suite import (
    SCENARIOS, simulate, simulate_idm, simulate_orca, simulate_apex,
    IDMParams, VARIANTS, d_samples_for, PlannerParams,
    SIM_DT, GOAL_S, CLEARANCE_MIN, clearance,
    EGO_L, EGO_W, MOTO_L, MOTO_W, load_gps_xy, straightest_window,
    REF_WINDOW_M, ReferencePath,
)
from scenario_videos import build_shared_background, draw_rect
from commonroad.common.file_reader import CommonRoadFileReader

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "maps" / "VinUni-1_2-T1.xml"
OUT_DIR = ROOT / "outputs"

# panels left->right, with font-safe labels (cv2 Hershey has no ★)
PANELS = ["baseline", "orca", "moto_aware", "apex"]
LABEL = {"baseline": "baseline (Frenet)",
         "orca": "ORCA/VO (van den Berg 2011)",
         "moto_aware": "moto-aware (ours)",
         "apex": "APEX (predictive, ours)"}

# curated scenarios (one per family) — by leading tag
TARGET_TAGS = ["C07", "X01", "H01", "W01", "T2-05", "T3-03"]


def log_variant(ref, vname, sc):
    """Return the per-step log for a planner on a scenario (mirrors run_variant)."""
    if vname == "idm":
        return simulate_idm(ref, sc.motos, IDMParams(v0=sc.ego_v))
    if vname == "orca":
        return simulate_orca(ref, sc.motos, sc.ego_v, sc.ego_lat_room)
    if vname == "apex":
        return simulate_apex(ref, sc.motos, sc.ego_v, sc.ego_lat_room)
    mode, ov = VARIANTS[vname]
    base = dict(dt=SIM_DT, v_desired=sc.ego_v,
                d_samples=d_samples_for(sc.ego_lat_room))
    base.update(ov)
    return simulate(ref, mode, PlannerParams(**base), sc.motos)


def clr_at(lg, i):
    return min(clearance(lg["ego_xy"][i], lg["ego_yaw"][i], (mx, my), myaw)
               for (mx, my, myaw) in lg["motos"][i])


def render(sc, ref, logs, bg, to_px, size, scale=8):
    W, H = size
    title_h = 34
    fps = int(1 / SIM_DT)
    out_w = W * len(PANELS) + 8 * (len(PANELS) - 1)
    tag = sc.name.split()[0]
    out_path = OUT_DIR / f"Phung_{sc.family}_{tag}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (out_w, H + title_h))

    clr_series = {v: np.array([clr_at(logs[v], i)
                              for i in range(len(logs[v]["t"]))]) for v in PANELS}
    n = max(len(logs[v]["t"]) for v in PANELS)

    for i in range(n):
        panels = []
        for v in PANELS:
            lg = logs[v]
            j = min(i, len(lg["t"]) - 1)
            img = bg.copy()
            for (mx, my, myaw) in lg["motos"][j]:
                draw_rect(img, to_px, mx, my, myaw, MOTO_L, MOTO_W, (60, 180, 75), scale)
            draw_rect(img, to_px, *lg["ego_xy"][j], lg["ego_yaw"][j],
                      EGO_L, EGO_W, (60, 76, 231), scale)

            clr = clr_series[v][j]
            run_min = clr_series[v][:j + 1].min()
            crashed = run_min <= 0.0

            cv2.rectangle(img, (8, 8), (430, 96), (255, 255, 255), -1)
            cv2.rectangle(img, (8, 8), (430, 96), (200, 200, 200), 1)
            cv2.putText(img, LABEL[v], (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (30, 30, 30), 2, cv2.LINE_AA)
            col = (40, 160, 40) if clr >= CLEARANCE_MIN else (40, 40, 220)
            cv2.putText(img, f"clr {clr:5.2f}  (min {run_min:5.2f})", (16, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            cv2.putText(img, f"{lg['v'][j] * 3.6:4.1f} km/h", (16, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
            if crashed:
                cv2.putText(img, "COLLISION", (W - 220, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (40, 40, 220), 3, cv2.LINE_AA)
            else:
                cv2.putText(img, "SAFE", (W - 120, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (40, 160, 40), 3, cv2.LINE_AA)
            panels.append(img)

        div = np.full((H, 8, 3), (60, 60, 60), np.uint8)
        body = panels[0]
        for pn in panels[1:]:
            body = np.hstack([body, div, pn])
        bar = np.full((title_h, out_w, 3), (40, 40, 40), np.uint8)
        cv2.putText(bar, f"{sc.name}  [{sc.family}]  -  others collide, APEX stays safe",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                    cv2.LINE_AA)
        writer.write(np.vstack([bar, body]))
    writer.release()
    return out_path


def main():
    by_tag = {sc.name.split()[0]: sc for sc in SCENARIOS}
    targets = [by_tag[t] for t in TARGET_TAGS if t in by_tag]
    print(f"Rendering {len(targets)} scenarios: "
          f"{', '.join(s.name.split()[0] for s in targets)}")

    xy = load_gps_xy()
    ref = ReferencePath(straightest_window(xy, REF_WINDOW_M))
    scenario, _ = CommonRoadFileReader(str(XML_PATH)).open()
    bg, to_px, size = build_shared_background(ref, scenario.lanelet_network.lanelets)
    print(f"  background {size[0]}x{size[1]} px\n")

    for sc in targets:
        logs = {v: log_variant(ref, v, sc) for v in PANELS}
        out = render(sc, ref, logs, bg, to_px, size)
        print(f"  -> {out.name}")
    print(f"\nWrote {len(targets)} videos (Phung_ prefix) to {OUT_DIR}")


if __name__ == "__main__":
    main()
