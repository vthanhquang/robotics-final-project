"""
Stress suite: 10 motorcycle lane-split / cut-in scenarios, baseline vs
motorcycle-aware planner.

Reuses the exact planner (src/frenet_planner.py) and the same Frenet-frame
scenario mechanics and PR2 metrics as src/planner_demo.py, but parameterises
the motorcycle script so we can sweep a battery of progressively harder /
more dangerous cut-ins. No map rendering -> fast and quiet (the CommonRoad
background is only needed for the side-by-side video).

Run:  python src/scenario_suite.py
Out:  outputs/scenario_suite.md   (also printed to stdout)
"""

import csv, math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from frenet_planner import PlannerParams, FrenetState, ObstacleState, ReferencePath, plan

# ── Paths (same data as planner_demo.py) ──────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
GPS_PATH = ROOT / "data" / "raw" / "sensors" / "Location.csv"
OUT_MD   = ROOT / "outputs" / "scenario_suite.md"
R_EARTH  = 6_378_137.0

# ── Vehicle geometry (pitch calibration table, same as planner_demo.py) ───────
EGO_L, EGO_W   = 4.5, 1.6
MOTO_L, MOTO_W = 2.0, 0.7

REF_WINDOW_M = 120.0
SIM_DT       = 0.1
SIM_T        = 18.0
GOAL_S       = 70.0

CLEARANCE_MIN = 0.3      # m, PR2 dense-traffic threshold
TTC_THRESH    = 2.0      # s
AEB_DECEL     = 3.0      # m/s^2


# ── Scenario definition ───────────────────────────────────────────────────────
@dataclass
class Scenario:
    name: str
    moto_v: float        # m/s longitudinal speed of moto
    moto_s0: float       # m, longitudinal gap ahead of ego at t=0
    moto_d0: float       # m, lateral offset before cut-in
    moto_d1: float       # m, lateral offset after cut-in (0 = ego lane centre)
    cut_t0: float        # s, cut-in start
    cut_t1: float        # s, cut-in end
    ego_v: float = 8.0   # m/s ego desired speed
    note: str = ""


# Ten escalating lane-split / cut-in scenarios. Knobs that make a cut-in
# dangerous: small initial gap, slow squeezing moto in front of a fast ego,
# short (snap) cut-in window, and late cut-in at close range.
SCENARIOS = [
    Scenario("S01 nominal cut-in",        4.0, 24.0, 2.6,  0.0, 2.5, 5.0,  8.0,
             "the planner_demo.py reference case"),
    Scenario("S02 close gap",             4.0, 16.0, 2.6,  0.0, 2.0, 4.0,  8.0,
             "starts only 16 m ahead"),
    Scenario("S03 slow squeeze, fast ego",3.0, 22.0, 2.6,  0.0, 2.5, 4.5, 10.0,
             "slow moto, ego wants 36 km/h"),
    Scenario("S04 snap cut-in",           4.0, 22.0, 2.8,  0.0, 2.0, 2.8,  8.0,
             "0.8 s lateral snap into lane"),
    Scenario("S05 late + close",          4.0, 14.0, 2.6,  0.0, 1.0, 2.5,  9.0,
             "cuts in early at 14 m, fast ego"),
    Scenario("S06 cross-over",            3.5, 20.0, 3.5, -0.3, 2.0, 4.0,  9.0,
             "comes from far side, overshoots centre"),
    Scenario("S07 high-speed closing",    5.0, 30.0, 2.6,  0.0, 2.5, 4.5, 12.0,
             "ego 43 km/h, larger closing rate"),
    Scenario("S08 stall-in",              2.0, 18.0, 2.6,  0.0, 2.0, 4.0,  9.0,
             "near-stationary moto plugs the lane"),
    Scenario("S09 aggressive close+snap", 3.5, 12.0, 2.6,  0.0, 0.8, 2.0, 10.0,
             "12 m, 1.2 s snap, fast ego"),
    Scenario("S10 worst case",            2.5, 10.0, 2.6,  0.0, 0.6, 1.6, 10.0,
             "10 m, 1.0 s snap, slow plug, fast ego"),
]


# ── GPS load + straightest window (same convention as planner_demo.py) ────────
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
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate([[0], np.cumsum(seg)])
    head = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    dhead = np.abs((np.diff(head) + np.pi) % (2 * np.pi) - np.pi)
    best_i, best_cost, best_j = 0, 1e18, len(xy) - 1
    for i in range(len(xy)):
        j = np.searchsorted(s, s[i] + window_m)
        if j >= len(xy):
            break
        cost = dhead[i:max(i + 1, j - 1)].sum()
        if cost < best_cost:
            best_cost, best_i, best_j = cost, i, j
    return xy[best_i:best_j + 1]


# ── Scripted motorcycle motion (parameterised smoothstep cut-in) ──────────────
def make_moto(sc: Scenario):
    span = max(sc.cut_t1 - sc.cut_t0, 1e-6)

    def moto_state(t):
        s = sc.moto_s0 + sc.moto_v * t
        if t <= sc.cut_t0:
            u, du = 0.0, 0.0
        elif t >= sc.cut_t1:
            u, du = 1.0, 0.0
        else:
            x = (t - sc.cut_t0) / span
            u = 3 * x ** 2 - 2 * x ** 3
            du = (6 * x - 6 * x ** 2) / span
        d = sc.moto_d0 + (sc.moto_d1 - sc.moto_d0) * u
        d_d = (sc.moto_d1 - sc.moto_d0) * du
        return ObstacleState(s=s, s_d=sc.moto_v, d=d, d_d=d_d)

    return moto_state


# ── Geometry: rectangle as discs, surface clearance (same as planner_demo.py) ─
def rect_discs(cx, cy, yaw, length, width, n):
    r = width / 2.0
    offs = np.linspace(-(length / 2 - r), (length / 2 - r), n)
    return [(cx + o * math.cos(yaw), cy + o * math.sin(yaw), r) for o in offs]


def clearance(ego_xy, ego_yaw, moto_xy, moto_yaw):
    eg = rect_discs(*ego_xy, ego_yaw, EGO_L, EGO_W, 3)
    mo = rect_discs(*moto_xy, moto_yaw, MOTO_L, MOTO_W, 2)
    return min(math.hypot(a[0] - b[0], a[1] - b[1]) - a[2] - b[2]
               for a in eg for b in mo)


# ── Run one planner mode over one scenario ────────────────────────────────────
def simulate(ref, mode, p, moto_state):
    ego = FrenetState(s=0.0, s_d=p.v_desired, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    log = {"t": [], "ego_xy": [], "ego_yaw": [], "v": [],
           "moto_xy": [], "moto_yaw": []}
    t = 0.0
    for _ in range(int(SIM_T / SIM_DT)):
        obs = moto_state(t)
        best, _ = plan(ego, obs, mode, p)

        ex, ey = ref.to_cartesian(ego.s, ego.d)
        ego_yaw = ref.yaw(ego.s) + math.atan2(ego.d_d, max(ego.s_d, 0.1))
        mx, my = ref.to_cartesian(obs.s, obs.d)
        moto_yaw = ref.yaw(obs.s) + math.atan2(obs.d_d, max(obs.s_d, 0.1))

        log["t"].append(t)
        log["ego_xy"].append((float(ex), float(ey)))
        log["ego_yaw"].append(ego_yaw)
        log["v"].append(ego.s_d)
        log["moto_xy"].append((float(mx), float(my)))
        log["moto_yaw"].append(moto_yaw)

        k = 1   # advance one SIM_DT along chosen trajectory
        ego = FrenetState(s=float(best.s[k]), s_d=float(best.v[k]),
                          s_dd=float(best.s_dd[k]), d=float(best.d[k]),
                          d_d=float(best.d_d[k]), d_dd=float(best.d_dd[k]))
        t += SIM_DT
        if ego.s >= GOAL_S:
            break
    return log


# ── Metrics (PR2 Table 3/4, same as planner_demo.py) ──────────────────────────
def metrics(log, human_speed):
    ego = np.array(log["ego_xy"]); moto = np.array(log["moto_xy"])
    v = np.array(log["v"]); t = np.array(log["t"])
    clr = np.array([clearance(log["ego_xy"][i], log["ego_yaw"][i],
                              log["moto_xy"][i], log["moto_yaw"][i])
                    for i in range(len(t))])
    rng = np.hypot(ego[:, 0] - moto[:, 0], ego[:, 1] - moto[:, 1])
    ttc = np.full_like(rng, np.inf)
    dr = np.gradient(rng, SIM_DT)
    closing = dr < -0.1
    ttc[closing] = rng[closing] / -dr[closing]
    ttc_exposure = float(np.mean((ttc > 0) & (ttc < TTC_THRESH))) * 100
    a = np.gradient(v, SIM_DT)
    hard = a < -AEB_DECEL
    aeb = int(np.sum(hard[1:] & ~hard[:-1]))
    dist = np.sum(np.hypot(np.diff(ego[:, 0]), np.diff(ego[:, 1])))
    t_total = t[-1] + SIM_DT
    rt = t_total / (dist / human_speed) if dist > 0 else float("nan")
    return {
        "min_clearance": float(clr.min()),
        "collision": int(np.sum(clr <= 0.0) > 0),
        "ttc_exposure": ttc_exposure,
        "aeb": aeb,
        "peak_decel": float(-a.min()),
        "min_speed": float(v.min()),
        "rt": float(rt),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading GPS + selecting straightest window...")
    xy = load_gps_xy()
    win = straightest_window(xy, REF_WINDOW_M)
    ref = ReferencePath(win)
    print(f"  reference path: {ref.length:.0f} m, {len(win)} pts\n")

    rows = []
    summary = {"baseline": 0, "moto_aware": 0}
    for sc in SCENARIOS:
        p = PlannerParams(dt=SIM_DT, v_desired=sc.ego_v)
        moto = make_moto(sc)
        res = {}
        for mode in ("baseline", "moto_aware"):
            res[mode] = metrics(simulate(ref, mode, p, moto), sc.ego_v)
        for mode in ("baseline", "moto_aware"):
            summary[mode] += res[mode]["collision"]
        rows.append((sc, res))
        b, m = res["baseline"], res["moto_aware"]
        print(f"{sc.name:28s} | baseline {'CRASH' if b['collision'] else ' ok  '} "
              f"clr {b['min_clearance']:6.2f} | moto-aware "
              f"{'CRASH' if m['collision'] else ' ok  '} clr {m['min_clearance']:6.2f}")

    # ── markdown report ──
    L = []
    L.append("# Motorcycle cut-in stress suite — baseline vs moto-aware planner")
    L.append("")
    L.append(f"Ten lane-split / cut-in scenarios on the real VinUni / Ocean Park "
             f"GPS reference path ({ref.length:.0f} m straightest window). Identical "
             f"scenario for both planners; only the cost function differs. "
             f"Collision = geometric clearance (rectangle footprints) <= 0; "
             f"min clearance threshold {CLEARANCE_MIN:.2f} m (PR2).")
    L.append("")
    L.append("| # | Scenario | Mode | Collision | Min clr (m) | TTC<2s (%) | "
             "Peak decel (m/s²) | Min spd (km/h) | R_T |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, (sc, res) in enumerate(rows, 1):
        for mode, label in (("baseline", "baseline"), ("moto_aware", "moto-aware")):
            r = res[mode]
            crash = "**YES**" if r["collision"] else "none"
            fail = "" if r["min_clearance"] >= CLEARANCE_MIN else " ⚠"
            L.append(f"| {i if mode=='baseline' else ''} "
                     f"| {sc.name if mode=='baseline' else ''} "
                     f"| {label} | {crash} | {r['min_clearance']:.2f}{fail} "
                     f"| {r['ttc_exposure']:.0f} | {r['peak_decel']:.1f} "
                     f"| {r['min_speed']*3.6:.1f} | {r['rt']:.2f} |")
    L.append("")
    L.append(f"**Collisions: baseline {summary['baseline']}/10, "
             f"moto-aware {summary['moto_aware']}/10.**")
    L.append("")
    L.append("Notes per scenario:")
    for i, (sc, _) in enumerate(rows, 1):
        L.append(f"- **{sc.name}** — {sc.note} "
                 f"(moto {sc.moto_v:.1f} m/s, gap {sc.moto_s0:.0f} m, "
                 f"cut {sc.cut_t0:.1f}–{sc.cut_t1:.1f} s, ego {sc.ego_v:.0f} m/s)")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\nCollisions: baseline {summary['baseline']}/10, "
          f"moto-aware {summary['moto_aware']}/10")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
