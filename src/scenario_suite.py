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
# Geometry grounded in the CommonRoad map (src/inspect_map.py): lane width
# LANE = 3.5 m near the route, the widest road there is 3 parallel lanes, and
# there are real perpendicular crossroads near the route start. Lateral offsets
# below are expressed in lanes: +LANE = one lane to the left of the ego lane
# centre, -LANE = one lane to the right (VN drives on the right).
LANE = 3.5          # m, measured median lane width near the route
HALF = LANE / 2.0   # m, lane edge / shoulder line


@dataclass
class Scenario:
    name: str
    family: str          # cut_in_left | cut_in_right | blocking | lane_split |
                         # crossing | multi_lane
    moto_v: float        # m/s longitudinal speed of moto (along ego's path)
    moto_s0: float       # m, longitudinal gap ahead of ego at t=0
    moto_d0: float       # m, lateral offset before the manoeuvre
    moto_d1: float       # m, lateral offset after the manoeuvre (0 = ego lane)
    cut_t0: float        # s, manoeuvre start
    cut_t1: float        # s, manoeuvre end
    ego_v: float = 8.0   # m/s ego desired speed
    ego_lat_room: float = 0.6  # m, lateral room the ego may use (road-dependent:
                               # 0.6 = single lane, ~LANE = multi-lane road)
    note: str = ""


# Realistic Vietnamese mixed-traffic motorcycle scenarios (deliberately NOT the
# earlier extreme/reckless set). Manoeuvres are gentle (3-5 s lane changes,
# 18-35 km/h motos, 18-32 m gaps); crossings are quick only because a crossing
# vehicle genuinely transits fast. Six families, weighted toward the right side
# (the common VN approach) plus middle-blocking, crossroad crossings and a
# 3-lane multi-lane change.
SCENARIOS = [
    # ── cut-in from the LEFT (gentle merge into ego lane) ──
    Scenario("L1 left cut-in, calm",      "cut_in_left",
             6.0, 28.0,  LANE, 0.0, 2.5, 6.0,  9.0, 0.6,
             "moto eases from the left lane over 3.5 s"),
    Scenario("L2 left cut-in, closer",    "cut_in_left",
             5.0, 22.0,  LANE, 0.0, 2.0, 5.5,  8.0, 0.6,
             "shorter 22 m gap, still a 3.5 s merge"),

    # ── cut-in from the RIGHT (more of these: common VN side) ──
    Scenario("R1 right cut-in, calm",     "cut_in_right",
             6.0, 28.0, -LANE, 0.0, 2.5, 6.0,  9.0, 0.6,
             "moto eases in from the right lane"),
    Scenario("R2 right cut-in, closer",   "cut_in_right",
             5.0, 22.0, -LANE, 0.0, 2.0, 5.0,  8.0, 0.6,
             "22 m gap, 3 s merge from the right"),
    Scenario("R3 right shoulder merge",   "cut_in_right",
             7.0, 32.0, -HALF, 0.0, 3.0, 6.5, 10.0, 0.6,
             "rides up the right shoulder then merges"),
    Scenario("R4 right cut-in, brisk",    "cut_in_right",
             5.5, 20.0, -LANE, 0.0, 2.0, 4.5,  9.0, 0.6,
             "slightly brisker 2.5 s merge from right"),

    # ── moto driving in the MIDDLE consistently until the car arrives ──
    Scenario("M1 slow lead in lane",      "blocking",
             4.0, 30.0,  0.0,  0.0, 0.0, 0.0,  9.0, 0.6,
             "moto holds ego lane at 14 km/h; ego catches up"),
    Scenario("M2 slower lead in lane",    "blocking",
             3.0, 24.0,  0.0,  0.0, 0.0, 0.0,  8.0, 0.6,
             "11 km/h lead, must follow / slow"),
    Scenario("M3 moderate lead",          "blocking",
             5.0, 26.0,  0.0,  0.0, 0.0, 0.0, 10.0, 0.6,
             "18 km/h lead vs a 36 km/h ego"),

    # ── lane-split: moto passes riding the lane line, partial encroach ──
    Scenario("S1 split on the left",      "lane_split",
             8.0, 12.0,  HALF, 1.0, 1.5, 4.0,  7.0, 0.6,
             "faster moto filters past on the left line"),
    Scenario("S2 split on the right",     "lane_split",
             8.0, 12.0, -HALF,-1.0, 1.5, 4.0,  7.0, 0.6,
             "faster moto filters past on the right line"),

    # ── crossing at a crossroad (real perpendicular crossing near route start) ──
    Scenario("X1 cross L->R at junction", "crossing",
             1.0, 18.0,  1.5*LANE, -1.5*LANE, 2.0, 3.4,  8.0, 0.6,
             "moto crosses ego path L->R at the crossroad"),
    Scenario("X2 cross R->L at junction", "crossing",
             1.0, 20.0, -1.5*LANE,  1.5*LANE, 2.5, 4.0,  9.0, 0.6,
             "moto crosses R->L, slightly later/faster ego"),
    Scenario("X3 hesitant crosser",       "crossing",
             1.0, 16.0,  1.3*LANE, -1.3*LANE, 2.0, 3.8,  7.0, 0.6,
             "slower crossing, closer to ego"),

    # ── multi-lane change on the 3-lane road (cross >1 lane); ego has room ──
    Scenario("W1 2-lane merge from left", "multi_lane",
             6.0, 26.0,  2*LANE, 0.0, 2.5, 7.0,  9.0, LANE,
             "moto crosses 2 lanes L->ego over 4.5 s; ego may shift"),
    Scenario("W2 2-lane merge from right","multi_lane",
             6.0, 26.0, -2*LANE, 0.0, 2.5, 7.0,  9.0, LANE,
             "moto crosses 2 lanes R->ego; ego may use the 3rd lane"),
    Scenario("W3 right drift to mid-lane","multi_lane",
             7.0, 30.0, -2*LANE,-LANE, 2.5, 6.5, 10.0, LANE,
             "moto moves 2 lanes -> adjacent lane (stops short of ego)"),
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


# ── Ego lateral sampling from the road's available room (map-grounded) ────────
def d_samples_for(room):
    return tuple(np.round(np.linspace(-room, room, 7), 2))


# ── Planner variants (incl. Phung's "conservative" model from his branch) ─────
# Each is (cost_mode, parameter overrides). "conservative" = baseline cost but
# big fixed buffers and a low desired speed ("just be cautious everywhere"),
# matching src/evaluate_planners.py on phung/planner-replay-tools.
VARIANTS = {
    "baseline":     ("baseline",   {}),
    "conservative": ("baseline",   dict(v_desired=5.0, base_lat_buf=2.6,
                                        base_long_buf=11.0)),
    "moto_aware":   ("moto_aware", {}),
}
VARIANT_ORDER = ["baseline", "conservative", "moto_aware"]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading GPS + selecting straightest window...")
    xy = load_gps_xy()
    win = straightest_window(xy, REF_WINDOW_M)
    ref = ReferencePath(win)
    print(f"  reference path: {ref.length:.0f} m, {len(win)} pts\n")

    rows = []
    n = len(SCENARIOS)
    summary = {v: 0 for v in VARIANT_ORDER}
    for sc in SCENARIOS:
        moto = make_moto(sc)
        res = {}
        for vname in VARIANT_ORDER:
            mode, ov = VARIANTS[vname]
            base = dict(dt=SIM_DT, v_desired=sc.ego_v,
                        d_samples=d_samples_for(sc.ego_lat_room))
            base.update(ov)
            res[vname] = metrics(simulate(ref, mode, PlannerParams(**base), moto),
                                 sc.ego_v)
            summary[vname] += res[vname]["collision"]
        rows.append((sc, res))
        cells = "  ".join(
            f"{v[:4]} {'X' if res[v]['collision'] else '.'}{res[v]['min_clearance']:5.2f}"
            for v in VARIANT_ORDER)
        print(f"{sc.name:26s} [{sc.family:12s}] | {cells}")

    # ── per-family aggregation ──
    families = []
    for sc, _ in rows:
        if sc.family not in families:
            families.append(sc.family)

    def agg(fam, vname, key):
        sub = [res[vname] for sc, res in rows if sc.family == fam]
        if key == "coll":
            return sum(r["collision"] for r in sub), len(sub)
        return float(np.mean([r[key] for r in sub]))

    # ── markdown report ──
    L = [f"# Realistic VN motorcycle scenario suite — 3 planners", "",
         f"{n} scenarios across {len(families)} behaviour families on the real "
         f"VinUni / Ocean Park GPS reference path ({ref.length:.0f} m straightest "
         f"window). Geometry grounded in the CommonRoad map: lane width "
         f"{LANE:.1f} m, up to 3 parallel lanes, real crossroads near the route "
         f"start. Manoeuvres are realistic (gentle 3–5 s merges, 18–35 km/h "
         f"motos). Three planners compared (Phung's *conservative* variant "
         f"included): **baseline**, **conservative** (big buffers + low speed), "
         f"**moto-aware** (ours). Collision = rectangle-footprint clearance ≤ 0; "
         f"pass threshold {CLEARANCE_MIN:.2f} m (PR2). R_T uses each scenario's "
         f"ego desired speed as the human reference.", "",
         "## Overall", "",
         "| Planner | Collisions | Mean min-clr | Mean R_T |",
         "|---|---|---|---|"]
    for v in VARIANT_ORDER:
        mclr = float(np.mean([res[v]["min_clearance"] for _, res in rows]))
        mrt = float(np.mean([res[v]["rt"] for _, res in rows]))
        L.append(f"| {v} | {summary[v]}/{n} | {mclr:.2f} m | {mrt:.2f} |")

    L += ["", "## By family (collisions | mean min-clearance)", "",
          "| Family | n | baseline | conservative | moto-aware |",
          "|---|---|---|---|---|"]
    for fam in families:
        cells = []
        for v in VARIANT_ORDER:
            c, nn = agg(fam, v, "coll")
            cells.append(f"{c}/{nn}, {agg(fam, v, 'min_clearance'):.2f} m")
        n_fam = agg(fam, "baseline", "coll")[1]
        L.append(f"| {fam} | {n_fam} | " + " | ".join(cells) + " |")

    L += ["", "## Per scenario", "",
          "| Scenario | Family | Planner | Collision | Min clr (m) | TTC<2s (%) | "
          "Peak decel (m/s²) | Min spd (km/h) | R_T |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sc, res in rows:
        for k, v in enumerate(VARIANT_ORDER):
            r = res[v]
            crash = "**YES**" if r["collision"] else "none"
            fail = "" if r["min_clearance"] >= CLEARANCE_MIN else " ⚠"
            L.append(f"| {sc.name if k==0 else ''} | {sc.family if k==0 else ''} "
                     f"| {v} | {crash} | {r['min_clearance']:.2f}{fail} "
                     f"| {r['ttc_exposure']:.0f} | {r['peak_decel']:.1f} "
                     f"| {r['min_speed']*3.6:.1f} | {r['rt']:.2f} |")

    L += ["", "**Overall collisions: " +
          ", ".join(f"{v} {summary[v]}/{n}" for v in VARIANT_ORDER) + ".**", "",
          "Scenario parameters:"]
    for sc, _ in rows:
        L.append(f"- **{sc.name}** ({sc.family}) — {sc.note} "
                 f"[moto {sc.moto_v:.1f} m/s, gap {sc.moto_s0:.0f} m, "
                 f"d {sc.moto_d0:+.1f}→{sc.moto_d1:+.1f} m, "
                 f"cut {sc.cut_t0:.1f}–{sc.cut_t1:.1f} s, ego {sc.ego_v:.0f} m/s, "
                 f"room ±{sc.ego_lat_room:.1f} m]")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\nOverall collisions: " +
          ", ".join(f"{v} {summary[v]}/{n}" for v in VARIANT_ORDER))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
