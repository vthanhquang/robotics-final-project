"""
Render one side-by-side (baseline | moto-aware) replay video per scenario for
the 10-case cut-in stress suite (src/scenario_suite.py).

Reuses the exact planner, scenarios and geometry; renders on the real VinUni /
Ocean Park CommonRoad map (loaded once, shared by all videos for a consistent
frame). Each video has the same HUD as planner_demo.py plus a CRASH banner.

Run:  python src/scenario_videos.py
Out:  outputs/scenario_videos/S01..S10.mp4
"""

import math
from pathlib import Path

import cv2
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader

from frenet_planner import PlannerParams, FrenetState, plan, ReferencePath
from scenario_suite import (
    SCENARIOS, make_moto, load_gps_xy, straightest_window, clearance,
    EGO_L, EGO_W, MOTO_L, MOTO_W, SIM_DT, SIM_T, GOAL_S, CLEARANCE_MIN,
    REF_WINDOW_M,
)

ROOT     = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "maps" / "VinUni-1_2-T1.xml"
OUT_DIR  = ROOT / "outputs" / "scenario_videos"


# ── Simulate one mode, logging everything the renderer needs ──────────────────
def simulate(ref, mode, p, moto_state):
    ego = FrenetState(s=0.0, s_d=p.v_desired, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    log = {"t": [], "ego_xy": [], "ego_yaw": [], "v": [], "moto_xy": [],
           "moto_yaw": [], "plan_xy": [], "cand_xy": [], "obs": []}
    t = 0.0
    for _ in range(int(SIM_T / SIM_DT)):
        obs = moto_state(t)
        best, cands = plan(ego, obs, mode, p)

        ex, ey = ref.to_cartesian(ego.s, ego.d)
        ego_yaw = ref.yaw(ego.s) + math.atan2(ego.d_d, max(ego.s_d, 0.1))
        mx, my = ref.to_cartesian(obs.s, obs.d)
        moto_yaw = ref.yaw(obs.s) + math.atan2(obs.d_d, max(obs.s_d, 0.1))
        bx, by = ref.to_cartesian(best.s, best.d)
        sub = cands[::max(1, len(cands) // 24)]
        cxy = [np.array(ref.to_cartesian(c.s, c.d)).T for c in sub]

        log["t"].append(t)
        log["ego_xy"].append((float(ex), float(ey)))
        log["ego_yaw"].append(ego_yaw)
        log["v"].append(ego.s_d)
        log["moto_xy"].append((float(mx), float(my)))
        log["moto_yaw"].append(moto_yaw)
        log["plan_xy"].append(np.array([bx, by]).T)
        log["cand_xy"].append(cxy)
        log["obs"].append(obs)

        k = 1
        ego = FrenetState(s=float(best.s[k]), s_d=float(best.v[k]),
                          s_dd=float(best.s_dd[k]), d=float(best.d[k]),
                          d_d=float(best.d_d[k]), d_dd=float(best.d_dd[k]))
        t += SIM_DT
        if ego.s >= GOAL_S:
            break
    return log


# ── Shared map background (built once for a consistent frame) ─────────────────
def build_shared_background(ref, lanelets, scale=8, pad=12):
    band = []
    for s in np.linspace(0, ref.length, 60):
        for d in (-6, 0, 6):
            band.append(ref.to_cartesian(s, d))
    xs = [p[0] for p in band]; ys = [p[1] for p in band]
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    W = int((xmax - xmin) * scale); H = int((ymax - ymin) * scale)

    def to_px(x, y):
        return int((x - xmin) * scale), int((ymax - y) * scale)

    bg = np.full((H, W, 3), (250, 248, 246), dtype=np.uint8)
    for ll in lanelets:
        v = ll.center_vertices
        if v[:, 0].max() < xmin or v[:, 0].min() > xmax or \
           v[:, 1].max() < ymin or v[:, 1].min() > ymax:
            continue
        poly = np.vstack([ll.left_vertices, ll.right_vertices[::-1]])
        cv2.fillPoly(bg, [np.array([to_px(*pp) for pp in poly], np.int32)],
                     (224, 224, 224))
        for vert in (ll.left_vertices, ll.right_vertices):
            cv2.polylines(bg, [np.array([to_px(*pp) for pp in vert], np.int32)],
                          False, (140, 140, 140), 2, cv2.LINE_AA)
    pts = [to_px(*ref.to_cartesian(s, 0)) for s in np.linspace(0, ref.length, 50)]
    cv2.polylines(bg, [np.array(pts, np.int32)], False, (180, 170, 150), 2,
                  cv2.LINE_AA)
    return bg, to_px, (W, H)


def draw_rect(img, to_px, cx, cy, yaw, L, Wd, color, scale):
    box = cv2.boxPoints(((0, 0), (L * scale, Wd * scale), -math.degrees(yaw)))
    px, py = to_px(cx, cy)
    box = (box + [px, py]).astype(np.int32)
    cv2.fillPoly(img, [box], color)
    cv2.polylines(img, [box], True, (40, 40, 40), 1, cv2.LINE_AA)


# ── Render one scenario to a side-by-side video ───────────────────────────────
def render_scenario(sc, ref, logs, p, bg, to_px, size, scale=8):
    W, H = size
    title_h = 34
    modes = [("baseline", "BASELINE", logs["baseline"]),
             ("moto_aware", "MOTO-AWARE", logs["moto_aware"])]
    n = min(len(logs["baseline"]["t"]), len(logs["moto_aware"]["t"]))
    fps = int(1 / SIM_DT)
    out_w = W * 2 + 8
    out_h = H + title_h
    out_path = OUT_DIR / f"{sc.name.split()[0]}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (out_w, out_h))

    # precompute running-min clearance per mode
    clr_series = {}
    for key, _, lg in modes:
        clr_series[key] = np.array(
            [clearance(lg["ego_xy"][i], lg["ego_yaw"][i],
                       lg["moto_xy"][i], lg["moto_yaw"][i]) for i in range(n)])

    for i in range(n):
        panels = []
        for key, title, lg in modes:
            img = bg.copy()
            for c in lg["cand_xy"][i]:
                pts = np.array([to_px(x, y) for x, y in c], np.int32)
                cv2.polylines(img, [pts], False, (205, 205, 205), 1, cv2.LINE_AA)
            pts = np.array([to_px(x, y) for x, y in lg["plan_xy"][i]], np.int32)
            cv2.polylines(img, [pts], False, (200, 120, 30), 2, cv2.LINE_AA)
            obs = lg["obs"][i]
            if key == "moto_aware" and abs(obs.d_d) > 0.05:
                lat = p.base_lat_buf + p.k_uncertainty * abs(obs.d_d) * 1.5
                mx, my = lg["moto_xy"][i]
                cv2.ellipse(img, to_px(mx, my),
                            (int(p.base_long_buf * scale), int(lat * scale)),
                            -math.degrees(lg["moto_yaw"][i]), 0, 360,
                            (150, 200, 150), 2, cv2.LINE_AA)
            draw_rect(img, to_px, *lg["moto_xy"][i], lg["moto_yaw"][i],
                      MOTO_L, MOTO_W, (60, 180, 75), scale)
            draw_rect(img, to_px, *lg["ego_xy"][i], lg["ego_yaw"][i],
                      EGO_L, EGO_W, (60, 76, 231), scale)

            clr = clr_series[key][i]
            run_min = clr_series[key][:i + 1].min()
            crashed = clr_series[key][:i + 1].min() <= 0.0

            cv2.rectangle(img, (8, 8), (360, 118), (255, 255, 255), -1)
            cv2.rectangle(img, (8, 8), (360, 118), (200, 200, 200), 1)
            cv2.putText(img, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (30, 30, 30), 2, cv2.LINE_AA)
            clr_col = (40, 160, 40) if clr >= CLEARANCE_MIN else (40, 40, 220)
            cv2.putText(img, f"clearance: {clr:5.2f} m  (min {run_min:5.2f})",
                        (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr_col, 2,
                        cv2.LINE_AA)
            cv2.putText(img, f"speed: {lg['v'][i] * 3.6:4.1f} km/h", (16, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1,
                        cv2.LINE_AA)
            cv2.putText(img, f"t = {lg['t'][i]:4.1f} s", (16, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                        cv2.LINE_AA)
            if crashed:
                cv2.putText(img, "COLLISION", (W - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 220), 3,
                            cv2.LINE_AA)
            panels.append(img)

        divider = np.full((H, 8, 3), (60, 60, 60), np.uint8)
        frame = np.hstack([panels[0], divider, panels[1]])
        title_bar = np.full((title_h, out_w, 3), (40, 40, 40), np.uint8)
        cv2.putText(title_bar, f"{sc.name}  -  {sc.note}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1,
                    cv2.LINE_AA)
        writer.write(np.vstack([title_bar, frame]))
    writer.release()
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading GPS + selecting straightest window...")
    xy = load_gps_xy()
    win = straightest_window(xy, REF_WINDOW_M)
    ref = ReferencePath(win)
    print(f"  reference path: {ref.length:.0f} m, {len(win)} pts")

    print("Loading CommonRoad map (once)...")
    scenario, _ = CommonRoadFileReader(str(XML_PATH)).open()
    lanelets = scenario.lanelet_network.lanelets
    bg, to_px, size = build_shared_background(ref, lanelets)
    print(f"  background {size[0]}x{size[1]} px\n")

    for sc in SCENARIOS:
        p = PlannerParams(dt=SIM_DT, v_desired=sc.ego_v)
        moto = make_moto(sc)
        logs = {m: simulate(ref, m, p, moto) for m in ("baseline", "moto_aware")}
        out = render_scenario(sc, ref, logs, p, bg, to_px, size)
        print(f"  -> {out.name}")
    print(f"\nWrote 10 videos to {OUT_DIR}")


if __name__ == "__main__":
    main()
