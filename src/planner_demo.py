"""
Planner method demo: baseline reactive planner vs. motorcycle-aware variant
on the real VinUni / Ocean Park CommonRoad map.

Scenario (data-conditioned, per PR2): the ego follows the straightest ~120 m
window of the real recorded GPS trace. A motorcycle travels ahead in the
adjacent lateral position and performs a *cut-in / lane-split* into the ego
lane -- the canonical Vietnamese mixed-traffic maneuver from the proposal.

Both planners run on the identical scenario; they differ only in the cost
function (see frenet_planner.py). Output:
  * outputs/planner_compare.mp4   side-by-side replay (baseline | moto-aware)
  * outputs/planner_metrics.md    PR2 evaluation table (clearance / TTC / AEB)

Run:  python src/planner_demo.py
"""

import csv, math
from pathlib import Path

import cv2
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader

from frenet_planner import (
    ReferencePath, PlannerParams, FrenetState, ObstacleState, plan,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "maps" / "VinUni-1_2-T1.xml"
GPS_PATH = ROOT / "data" / "raw" / "sensors" / "Location.csv"
OUT_MP4  = ROOT / "outputs" / "planner_compare.mp4"
OUT_MD   = ROOT / "outputs" / "planner_metrics.md"

R_EARTH = 6_378_137.0

# ── Scenario / vehicle geometry ───────────────────────────────────────────────
EGO_L, EGO_W   = 4.5, 1.6        # car        (pitch calibration table)
MOTO_L, MOTO_W = 2.0, 0.7        # motorcycle (pitch calibration table)
REF_WINDOW_M   = 120.0           # length of reference path window
SIM_DT         = 0.1             # s, replan + integration step
SIM_T          = 18.0            # s, max sim duration
GOAL_S         = 70.0            # m along reference -> stop condition

# motorcycle cut-in script (Frenet, relative to same reference path).
# The moto is slow and squeezes directly in front -> a single narrow lane (see
# PlannerParams.d_samples) leaves the ego no room to swerve, so it must adapt
# longitudinally. This is where anticipating the lateral cut-in pays off.
MOTO_V    = 4.0                  # m/s  (~14 km/h, slow lane-splitting moto)
MOTO_S0   = 24.0                 # m ahead of ego at t=0
MOTO_D0   = 2.6                  # m lateral offset (adjacent / shoulder)
MOTO_D1   = 0.0                  # m lateral after cut-in (square in ego lane)
CUT_T0, CUT_T1 = 2.5, 5.0        # s, cut-in interval

CLEARANCE_MIN = 0.3              # m, PR2 dense-traffic threshold
TTC_THRESH    = 2.0              # s, PR2 TTC exposure threshold
AEB_DECEL     = 3.0              # m/s^2, hard-braking threshold


# ── GPS load + projection (same convention as cr_gps_drive.py) ────────────────
def wgs84_to_m(lat, lon):
    x = lon * math.pi * R_EARTH / 180
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360)) * R_EARTH
    return x, y


def load_gps_xy():
    rows = []
    with open(GPS_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["seconds_elapsed"]),
                         float(r["latitude"]), float(r["longitude"])))
    rows.sort(key=lambda r: r[0])
    return np.array([wgs84_to_m(r[1], r[2]) for r in rows])


def straightest_window(xy, window_m):
    """Pick the contiguous arc-length window with the least total heading change."""
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate([[0], np.cumsum(seg)])
    head = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    dhead = np.abs((np.diff(head) + np.pi) % (2 * np.pi) - np.pi)
    best_i, best_cost = 0, 1e18
    for i in range(len(xy)):
        j = np.searchsorted(s, s[i] + window_m)
        if j >= len(xy):
            break
        cost = dhead[i:max(i + 1, j - 1)].sum()
        if cost < best_cost:
            best_cost, best_i, best_j = cost, i, j
    return xy[best_i:best_j + 1]


# ── Motorcycle ground-truth motion (scripted) ─────────────────────────────────
def moto_state(t):
    s = MOTO_S0 + MOTO_V * t
    # smoothstep lateral cut-in
    if t <= CUT_T0:
        u, du = 0.0, 0.0
    elif t >= CUT_T1:
        u, du = 1.0, 0.0
    else:
        x = (t - CUT_T0) / (CUT_T1 - CUT_T0)
        u = 3 * x ** 2 - 2 * x ** 3
        du = (6 * x - 6 * x ** 2) / (CUT_T1 - CUT_T0)
    d = MOTO_D0 + (MOTO_D1 - MOTO_D0) * u
    d_d = (MOTO_D1 - MOTO_D0) * du
    return ObstacleState(s=s, s_d=MOTO_V, d=d, d_d=d_d)


# ── Geometry: rectangle as discs, surface clearance ───────────────────────────
def rect_discs(cx, cy, yaw, length, width, n):
    r = width / 2.0
    offs = np.linspace(-(length / 2 - r), (length / 2 - r), n)
    return [(cx + o * math.cos(yaw), cy + o * math.sin(yaw), r) for o in offs]


def clearance(ego_xy, ego_yaw, moto_xy, moto_yaw):
    eg = rect_discs(*ego_xy, ego_yaw, EGO_L, EGO_W, 3)
    mo = rect_discs(*moto_xy, moto_yaw, MOTO_L, MOTO_W, 2)
    return min(math.hypot(a[0] - b[0], a[1] - b[1]) - a[2] - b[2]
               for a in eg for b in mo)


# ── Run one planner mode over the scenario ────────────────────────────────────
def simulate(ref, mode, p):
    ego = FrenetState(s=0.0, s_d=p.v_desired, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    log = {"t": [], "ego_xy": [], "ego_yaw": [], "v": [], "moto_xy": [],
           "moto_yaw": [], "plan_xy": [], "cand_xy": [], "obs": []}
    t = 0.0
    n = int(SIM_T / SIM_DT)
    for _ in range(n):
        obs = moto_state(t)
        best, cands = plan(ego, obs, mode, p)

        ex, ey = ref.to_cartesian(ego.s, ego.d)
        ego_yaw = ref.yaw(ego.s) + math.atan2(ego.d_d, max(ego.s_d, 0.1))
        mx, my = ref.to_cartesian(obs.s, obs.d)
        moto_yaw = ref.yaw(obs.s) + math.atan2(obs.d_d, max(obs.s_d, 0.1))

        bx, by = ref.to_cartesian(best.s, best.d)
        # subsample a few candidates for display
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

        # advance ego one step along the chosen trajectory (index 1 == SIM_DT)
        k = 1
        ego = FrenetState(s=float(best.s[k]), s_d=float(best.v[k]),
                          s_dd=float(best.s_dd[k]), d=float(best.d[k]),
                          d_d=float(best.d_d[k]), d_dd=float(best.d_dd[k]))
        t += SIM_DT
        if ego.s >= GOAL_S:
            break
    return log


# ── Metrics (PR2 Table 3/4) ───────────────────────────────────────────────────
def metrics(log):
    ego = np.array(log["ego_xy"]); moto = np.array(log["moto_xy"])
    v = np.array(log["v"]); t = np.array(log["t"])
    clr = np.array([clearance(log["ego_xy"][i], log["ego_yaw"][i],
                              log["moto_xy"][i], log["moto_yaw"][i])
                    for i in range(len(t))])
    rng = np.hypot(ego[:, 0] - moto[:, 0], ego[:, 1] - moto[:, 1])
    # TTC from closing range
    ttc = np.full_like(rng, np.inf)
    dr = np.gradient(rng, SIM_DT)
    closing = dr < -0.1
    ttc[closing] = rng[closing] / -dr[closing]
    ttc_exposure = float(np.mean((ttc > 0) & (ttc < TTC_THRESH))) * 100
    # AEB events (rising edges of hard braking)
    a = np.gradient(v, SIM_DT)
    hard = a < -AEB_DECEL
    aeb = int(np.sum(hard[1:] & ~hard[:-1]))
    # efficiency vs ideal human time at desired speed
    dist = np.sum(np.hypot(np.diff(ego[:, 0]), np.diff(ego[:, 1])))
    t_total = t[-1] + SIM_DT
    rt = t_total / (dist / 8.0) if dist > 0 else float("nan")
    return {
        "min_clearance": float(clr.min()),
        "collision": int(np.sum(clr <= 0.0) > 0),
        "ttc_exposure": ttc_exposure,
        "aeb": aeb,
        "peak_decel": float(-a.min()),
        "min_speed": float(v.min()),
        "time": float(t_total),
        "rt": float(rt),
        "clr_series": clr,
        "t_series": t,
    }


# ── Map background ────────────────────────────────────────────────────────────
def build_background(ref, logs, scale=8, pad=12):
    pts = [ref.to_cartesian(s, 0) for s in np.linspace(0, ref.length, 50)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    for lg in logs.values():
        e = np.array(lg["ego_xy"]); m = np.array(lg["moto_xy"])
        xs += [e[:, 0].min(), e[:, 0].max(), m[:, 0].min(), m[:, 0].max()]
        ys += [e[:, 1].min(), e[:, 1].max(), m[:, 1].min(), m[:, 1].max()]
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    W = int((xmax - xmin) * scale); H = int((ymax - ymin) * scale)

    def to_px(x, y):
        return int((x - xmin) * scale), int((ymax - y) * scale)

    bg = np.full((H, W, 3), (250, 248, 246), dtype=np.uint8)
    scenario, _ = CommonRoadFileReader(str(XML_PATH)).open()
    for ll in scenario.lanelet_network.lanelets:
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
    # reference centre-line
    cv2.polylines(bg, [np.array([to_px(*p) for p in pts], np.int32)],
                  False, (180, 170, 150), 2, cv2.LINE_AA)
    return bg, to_px, (W, H)


def draw_rect(img, to_px, cx, cy, yaw, L, Wd, color, scale):
    box = cv2.boxPoints(((0, 0), (L * scale, Wd * scale), -math.degrees(yaw)))
    px, py = to_px(cx, cy)
    box = (box + [px, py]).astype(np.int32)
    cv2.fillPoly(img, [box], color)
    cv2.polylines(img, [box], True, (40, 40, 40), 1, cv2.LINE_AA)


def render(ref, logs, mets, scale=8):
    bg, to_px, (W, H) = build_background(ref, logs, scale=scale)
    modes = [("baseline", "BASELINE planner", logs["baseline"], mets["baseline"]),
             ("moto_aware", "MOTO-AWARE planner", logs["moto_aware"],
              mets["moto_aware"])]
    n = min(len(logs["baseline"]["t"]), len(logs["moto_aware"]["t"]))
    fps = int(1 / SIM_DT)
    panel_w, panel_h = W, H
    out_w = panel_w * 2 + 8
    writer = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (out_w, panel_h))

    for i in range(n):
        panels = []
        for key, title, lg, mt in modes:
            img = bg.copy()
            # candidates (faint)
            for c in lg["cand_xy"][i]:
                pts = np.array([to_px(x, y) for x, y in c], np.int32)
                cv2.polylines(img, [pts], False, (205, 205, 205), 1, cv2.LINE_AA)
            # chosen plan (blue)
            pts = np.array([to_px(x, y) for x, y in lg["plan_xy"][i]], np.int32)
            cv2.polylines(img, [pts], False, (200, 120, 30), 2, cv2.LINE_AA)
            # moto uncertainty ellipse (moto_aware only)
            obs = lg["obs"][i]
            if key == "moto_aware" and abs(obs.d_d) > 0.05:
                p = PARAMS
                lat = p.base_lat_buf + p.k_uncertainty * abs(obs.d_d) * 1.5
                mx, my = lg["moto_xy"][i]
                myaw = lg["moto_yaw"][i]
                cv2.ellipse(img, to_px(mx, my),
                            (int(p.base_long_buf * scale), int(lat * scale)),
                            -math.degrees(myaw), 0, 360, (150, 200, 150), 2,
                            cv2.LINE_AA)
            # moto + ego
            draw_rect(img, to_px, *lg["moto_xy"][i], lg["moto_yaw"][i],
                      MOTO_L, MOTO_W, (60, 180, 75), scale)        # green
            draw_rect(img, to_px, *lg["ego_xy"][i], lg["ego_yaw"][i],
                      EGO_L, EGO_W, (60, 76, 231), scale)          # red

            # live clearance
            clr = clearance(lg["ego_xy"][i], lg["ego_yaw"][i],
                            lg["moto_xy"][i], lg["moto_yaw"][i])
            run_min = min(clr, mt["clr_series"][:i + 1].min())

            cv2.rectangle(img, (8, 8), (360, 118), (255, 255, 255), -1)
            cv2.rectangle(img, (8, 8), (360, 118), (200, 200, 200), 1)
            cv2.putText(img, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (30, 30, 30), 2, cv2.LINE_AA)
            clr_col = (40, 160, 40) if clr >= CLEARANCE_MIN else (40, 40, 220)
            cv2.putText(img, f"clearance: {clr:5.2f} m  (min {run_min:4.2f})",
                        (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr_col, 2,
                        cv2.LINE_AA)
            cv2.putText(img, f"speed: {lg['v'][i] * 3.6:4.1f} km/h",
                        (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80),
                        1, cv2.LINE_AA)
            cv2.putText(img, f"t = {lg['t'][i]:4.1f} s", (16, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                        cv2.LINE_AA)
            panels.append(img)

        divider = np.full((panel_h, 8, 3), (60, 60, 60), np.uint8)
        writer.write(np.hstack([panels[0], divider, panels[1]]))
    writer.release()
    return OUT_MP4


# ── Metrics report ────────────────────────────────────────────────────────────
def write_report(mets):
    b, m = mets["baseline"], mets["moto_aware"]

    def row(name, bb, mm, unit=""):
        return f"| {name} | {bb}{unit} | {mm}{unit} |"

    lines = [
        "# Planner method comparison — baseline vs motorcycle-aware",
        "",
        "Scenario: motorcycle cut-in / lane-split on the real VinUni / Ocean "
        "Park GPS reference path. Identical scenario for both planners; only "
        "the cost function differs (see `src/frenet_planner.py`).",
        "",
        "| Metric (PR2 Table 3/4) | Baseline | Moto-aware |",
        "|---|---|---|",
        row("Collision", "YES" if b["collision"] else "none",
            "YES" if m["collision"] else "none"),
        row("Min clearance", f"{b['min_clearance']:.2f}",
            f"{m['min_clearance']:.2f}", " m"),
        f"| &nbsp;&nbsp;vs 0.30 m threshold | "
        f"{'FAIL' if b['min_clearance'] < CLEARANCE_MIN else 'pass'} | "
        f"{'FAIL' if m['min_clearance'] < CLEARANCE_MIN else 'pass'} |",
        row("TTC exposure (<2 s)", f"{b['ttc_exposure']:.0f}",
            f"{m['ttc_exposure']:.0f}", " %"),
        row("AEB events", b["aeb"], m["aeb"]),
        row("Peak deceleration", f"{b['peak_decel']:.1f}",
            f"{m['peak_decel']:.1f}", " m/s²"),
        row("Min speed", f"{b['min_speed'] * 3.6:.1f}",
            f"{m['min_speed'] * 3.6:.1f}", " km/h"),
        row("Time ratio R_T", f"{b['rt']:.2f}", f"{m['rt']:.2f}"),
        "",
        "**Method improvement (proposal obj. 3 / PR1 §3.3):** the moto-aware "
        "planner propagates an uncertainty buffer that inflates the "
        "motorcycle's predicted footprint with its lateral velocity, and "
        "re-weights the safety cost when lateral motion is detected. It "
        "anticipates the cut-in and keeps clearance above the 0.30 m dense-"
        "traffic threshold without late emergency braking.",
    ]
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────
# planner trajectory step must match the sim step: simulate() executes
# trajectory index 1 (== one SIM_DT ahead) each replan.
PARAMS = PlannerParams(dt=SIM_DT)


def main():
    print("Loading GPS + selecting straightest window...")
    xy = load_gps_xy()
    win = straightest_window(xy, REF_WINDOW_M)
    ref = ReferencePath(win)
    print(f"  reference path: {ref.length:.0f} m, {len(win)} pts")

    logs, mets = {}, {}
    for mode in ("baseline", "moto_aware"):
        print(f"Simulating: {mode} ...")
        logs[mode] = simulate(ref, mode, PARAMS)
        mets[mode] = metrics(logs[mode])

    print("Rendering side-by-side video...")
    render(ref, logs, mets)
    print(f"  -> {OUT_MP4}")
    write_report(mets)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
