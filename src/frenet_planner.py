"""
Frenet-frame sampling reactive planner (Werling et al.) with a baseline and a
motorcycle-aware cost variant.

This is the *method* the proposal promised (pitch obj. 3 / PR1 sec. 3.2-3.3):
a sampling-based reactive planner that decouples motion into the curvilinear
Frenet frame (s, d), generates candidate trajectories with quintic/quartic
polynomials (C2-continuous), scores them, and picks the lowest-cost feasible
one in a receding horizon.

Two cost modes are provided so the planner's improvement over the proposal
baseline can be measured directly:

  * "baseline"  - classic Werling cost. Obstacles are predicted with constant
                  velocity and a *fixed* safety buffer; lateral intent (a
                  motorcycle cutting in) is not anticipated.

  * "moto_aware" - the PR1 sec. 3.3 contribution:
                  (1) Uncertainty-buffer propagation: the predicted occupied
                      area of a motorcycle is inflated laterally in proportion
                      to its lateral velocity and the prediction horizon, so a
                      lane-split / cut-in is anticipated before it happens.
                  (2) Dynamic cost re-weighting: the safety weight w_safe is
                      scaled up when the obstacle shows high lateral velocity
                      (a proxy for the "high heading variance" signal in PR1).

Self-contained: needs only numpy (+ scipy if available for a smoother spline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:                                            # smoother reference if available
    from scipy.interpolate import CubicSpline
    _HAVE_SCIPY = True
except Exception:                               # pragma: no cover
    _HAVE_SCIPY = False


# ── Polynomials ───────────────────────────────────────────────────────────────
class QuinticPolynomial:
    """Position-to-position with full boundary (pos, vel, acc) at both ends."""

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        self.a0, self.a1, self.a2 = xs, vxs, axs / 2.0
        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T**4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xe - (self.a0 + self.a1*T + self.a2*T**2),
                      vxe - (self.a1 + 2*self.a2*T),
                      axe - 2*self.a2])
        self.a3, self.a4, self.a5 = np.linalg.solve(A, b)

    def calc(self, t):
        return (self.a0 + self.a1*t + self.a2*t**2
                + self.a3*t**3 + self.a4*t**4 + self.a5*t**5)

    def calc_d(self, t):
        return (self.a1 + 2*self.a2*t + 3*self.a3*t**2
                + 4*self.a4*t**3 + 5*self.a5*t**4)

    def calc_dd(self, t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def calc_ddd(self, t):
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2


class QuarticPolynomial:
    """Velocity-keeping: end position free, end (vel, acc) fixed."""

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        self.a0, self.a1, self.a2 = xs, vxs, axs / 2.0
        A = np.array([[3*T**2, 4*T**3],
                      [6*T, 12*T**2]])
        b = np.array([vxe - (self.a1 + 2*self.a2*T),
                      axe - 2*self.a2])
        self.a3, self.a4 = np.linalg.solve(A, b)

    def calc(self, t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4

    def calc_d(self, t):
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3

    def calc_dd(self, t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2

    def calc_ddd(self, t):
        return 6*self.a3 + 24*self.a4*t


# ── Reference path (arc-length parameterised) ─────────────────────────────────
class ReferencePath:
    """Smooth centre-line with Cartesian <-> Frenet conversion."""

    def __init__(self, xy: np.ndarray):
        xy = np.asarray(xy, float)
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        if _HAVE_SCIPY:
            self._fx = CubicSpline(self.s, xy[:, 0])
            self._fy = CubicSpline(self.s, xy[:, 1])
        else:                                   # smoothed linear fallback
            self._xy = xy
        # dense lookup table for fast nearest-point projection
        self._ss = np.linspace(0, self.length, max(400, int(self.length * 4)))
        self._px = self.x(self._ss)
        self._py = self.y(self._ss)

    def x(self, s):
        if _HAVE_SCIPY:
            return self._fx(np.clip(s, 0, self.length))
        return np.interp(s, self.s, self._xy[:, 0])

    def y(self, s):
        if _HAVE_SCIPY:
            return self._fy(np.clip(s, 0, self.length))
        return np.interp(s, self.s, self._xy[:, 1])

    def yaw(self, s):
        ds = 0.1
        s0 = np.clip(s, 0, self.length - ds)
        return np.arctan2(self.y(s0 + ds) - self.y(s0),
                          self.x(s0 + ds) - self.x(s0))

    def to_cartesian(self, s, d):
        th = self.yaw(s)
        return (self.x(s) - d * np.sin(th),    # +d = left of travel direction
                self.y(s) + d * np.cos(th))

    def to_frenet(self, x, y):
        i = int(np.argmin((self._px - x) ** 2 + (self._py - y) ** 2))
        s = self._ss[i]
        th = self.yaw(s)
        dx, dy = x - self._px[i], y - self._py[i]
        d = -np.sin(th) * dx + np.cos(th) * dy
        return s, d


# ── Parameters & states ───────────────────────────────────────────────────────
@dataclass
class PlannerParams:
    dt: float = 0.2                     # trajectory sample step [s]
    horizons: tuple = (3.0, 4.0, 5.0)   # planning horizons T [s]
    # narrow single lane: the ego cannot dodge a cut-in laterally, so it must
    # respond by adapting speed (where anticipation matters).
    d_samples: tuple = tuple(np.round(np.linspace(-0.6, 0.6, 7), 2))
    v_samples: int = 6                  # target-speed samples below desired
    v_desired: float = 8.0             # [m/s] ~29 km/h
    v_step: float = 1.5                 # spacing of target-speed samples

    # cost weights
    w_jerk: float = 0.10
    w_time: float = 0.10
    w_lat: float = 1.0
    w_speed: float = 0.6
    w_safe: float = 6.0

    # safety / collision geometry
    base_long_buf: float = 7.0          # [m] longitudinal safety buffer
    base_lat_buf: float = 1.6           # [m] lateral safety buffer (baseline)
    k_uncertainty: float = 1.6          # lateral inflation per (m/s)*s  (moto_aware)
    beta_reweight: float = 1.8          # w_safe gain per (m/s) lateral vel (moto_aware)
    react_lat_thresh: float = 0.15      # [m/s] lateral vel that triggers re-weighting

    # hard collision footprint (>= combined vehicle half-lengths / -widths so a
    # flagged collision corresponds to real geometric overlap).
    hard_long: float = 3.4              # [m] longitudinal centre-gap
    hard_lat: float = 1.1              # [m] lateral centre-gap

    a_max: float = 3.0                  # feasibility: |long accel| limit [m/s^2]


@dataclass
class FrenetState:
    s: float
    s_d: float
    s_dd: float = 0.0
    d: float = 0.0
    d_d: float = 0.0
    d_dd: float = 0.0


@dataclass
class ObstacleState:
    """Current Frenet state of a tracked obstacle (e.g. a motorcycle)."""
    s: float
    s_d: float
    d: float
    d_d: float = 0.0      # lateral velocity  (the cut-in / lane-split signal)


@dataclass
class Candidate:
    t: np.ndarray
    s: np.ndarray
    d: np.ndarray
    v: np.ndarray            # s_d  (longitudinal speed)
    s_dd: np.ndarray
    d_d: np.ndarray
    d_dd: np.ndarray
    cost: float
    cost_safe: float
    feasible: bool = True


# ── Planner ───────────────────────────────────────────────────────────────────
def _predict_obstacle(obs: ObstacleState, t: np.ndarray, mode: str,
                      p: PlannerParams):
    """Return predicted (s, d, lat_buffer) arrays over horizon t."""
    s_pred = obs.s + obs.s_d * t
    if mode == "moto_aware":
        # (1) uncertainty-buffer propagation: lateral footprint grows with the
        #     observed lateral velocity and how far ahead we look.
        d_pred = obs.d + obs.d_d * t           # anticipate the lateral motion
        lat_buf = p.base_lat_buf + p.k_uncertainty * abs(obs.d_d) * t
    else:
        # baseline: assume the obstacle holds its current lateral position.
        d_pred = np.full_like(t, obs.d)
        lat_buf = np.full_like(t, p.base_lat_buf)
    return s_pred, d_pred, lat_buf


def plan(ego: FrenetState, obs: ObstacleState, mode: str, p: PlannerParams):
    """One receding-horizon planning step. Returns (best, all_candidates)."""
    # dynamic safety re-weighting (moto_aware only)
    w_safe = p.w_safe
    if mode == "moto_aware" and abs(obs.d_d) > p.react_lat_thresh:
        w_safe *= (1.0 + p.beta_reweight * abs(obs.d_d))

    v_targets = [max(0.0, p.v_desired - k * p.v_step) for k in range(p.v_samples)]

    candidates: list[Candidate] = []
    for T in p.horizons:
        t = np.arange(0.0, T + 1e-9, p.dt)
        s_o, d_o, lat_buf = _predict_obstacle(obs, t, mode, p)
        for d1 in p.d_samples:
            lat = QuinticPolynomial(ego.d, ego.d_d, ego.d_dd, d1, 0.0, 0.0, T)
            d = lat.calc(t)
            d_d = lat.calc_d(t)
            d_dd = lat.calc_dd(t)
            d_ddd = lat.calc_ddd(t)
            for v1 in v_targets:
                lon = QuarticPolynomial(ego.s, ego.s_d, ego.s_dd, v1, 0.0, T)
                s = lon.calc(t)
                s_d = lon.calc_d(t)
                s_dd = lon.calc_dd(t)
                s_ddd = lon.calc_ddd(t)

                feasible = np.all(np.abs(s_dd) <= p.a_max + 2.0)

                # costs (Werling): jerk + time + lateral + speed + safety
                Jp = np.sum(d_ddd ** 2)
                Js = np.sum(s_ddd ** 2)
                c_jerk = p.w_jerk * (Jp + Js)
                c_time = p.w_time * T
                c_lat = p.w_lat * d1 ** 2
                c_speed = p.w_speed * (p.v_desired - v1) ** 2

                # safety: elliptical proximity to the predicted obstacle footprint
                ds = s - s_o
                dd = d - d_o
                overlap = 1.0 - (ds / p.base_long_buf) ** 2 - (dd / lat_buf) ** 2
                overlap = np.clip(overlap, 0.0, None)
                c_safe = w_safe * float(np.sum(overlap ** 2))
                # hard collision (geometric) makes the candidate infeasible
                if np.any((np.abs(ds) < p.hard_long) & (np.abs(dd) < p.hard_lat)):
                    feasible = False

                cost = c_jerk + c_time + c_lat + c_speed + c_safe
                candidates.append(Candidate(t, s, d, s_d, s_dd, d_d, d_dd,
                                            cost, c_safe, feasible))

    feas = [c for c in candidates if c.feasible]
    if feas:
        best = min(feas, key=lambda c: c.cost)
    else:
        # no collision-free option: brace, don't plow through. Pick the safest
        # (least predicted overlap), tie-broken by lowest speed.
        best = min(candidates, key=lambda c: (c.cost_safe, c.v[-1]))
    return best, candidates
