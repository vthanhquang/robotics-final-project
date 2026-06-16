# Vietnam-MixedTrafficSim — Final Project Report

**ELEC5050 Robotics — Project Group 15, VinUniversity**

Do Minh Phung (V202603145) · Vu Thanh Quang (V202603120) · Nguyen Quoc Linh (V202603142)

> Reproduce every number in this report with `python src/make_demo.py`
> (regenerates the videos, the metrics, the slides and the reel in ~2.5 min).
> The authoritative metrics tables are written to `outputs/planner_metrics.md`
> and `outputs/planner_evaluation.md`.

---

## Abstract

Autonomous-driving motion planners are designed for Western traffic — marked
lanes and rule-abiding drivers. Vietnamese roads are motorcycle-dominated
(>90% of vehicles), where two-wheelers lane-split and cut in at sub-meter range.
We build a simulation-first pipeline on the **real VinUniversity / Vinhomes
Ocean Park road network** and a recorded drive, and we implement and evaluate a
**motorcycle-aware reactive planner**. On a 24-scenario suite the motorcycle-aware
cost function reduces the collision rate from **38% to 12%** and raises the
clearance-pass rate from **50% to 88%**, *while keeping the same efficiency as
the baseline when no real conflict is present* (time ratio R_T = 1.01 in both).
This shows the improvement is targeted, not merely cautious.

---

## 1. Problem and motivation (CLO4)

Standard planners assume obstacles keep their lane; a motorcycle cutting in
laterally is therefore reacted to too late. In Vietnam this is the dominant
hazard. Our goal: a planner that **anticipates** motorcycle cut-ins on the real
local road, evaluated with quantitative safety/efficiency metrics.

## 2. Related work

CommonRoad provides the benchmark format and the reactive-planner formulation
(Werling et al.); CARLA provides 3D simulation; SUMO provides microscopic
traffic with a sub-lane model for two-wheelers. Existing platforms each miss at
least one of {OSM map, motorcycle agent, Lanelet2, planner-evaluation loop}.
Our contribution is the motorcycle-aware planner and its evaluation on a real
VN road. (Full reference list in the pitch and progress reports.)

## 3. System overview

1. **Perception + replay** — YOLOv8 obstacle detection with monocular distance
   on the dashcam video; GPS replayed on the CommonRoad/Lanelet2 map; a drive-
   stats panel; composited into a PR2-style 3-panel replay.
2. **Planning** — a Frenet-frame sampling reactive planner with a baseline and a
   motorcycle-aware cost variant.
3. **Evaluation** — PR2 safety/efficiency metrics over a scenario suite.

Map: `VinUni-1_2-T1.xml` (4797 lanelets). Drive: ~220 s, ~1.49 km, peak
~42.7 km/h, iPhone 12 Sensor Logger (GPS ~1 Hz, IMU ~100 Hz).

## 4. Method

### 4.1 Frenet-frame reactive planner

Motion is decoupled into the curvilinear Frenet frame (s, d) about a reference
path. Candidate trajectories are quintic (lateral) and quartic (longitudinal,
velocity-keeping) polynomials, giving C²-continuity. Each candidate is scored

> J = w_jerk·J_jerk + w_time·T + w_lat·d₁² + w_speed·(v_des − v₁)² + w_safe·C_safe

and the lowest-cost *feasible* candidate is executed in a receding horizon
(replan every 0.1 s). Implementation: `src/frenet_planner.py`.

### 4.2 Motorcycle-aware cost (our contribution)

Two changes over the baseline (proposal obj. 3 / PR1 §3.3):

1. **Uncertainty-buffer propagation.** The motorcycle's predicted occupied
   region is inflated laterally in proportion to its lateral velocity and the
   prediction horizon: `lat_buf(t) = base + k·|ḋ_moto|·t`. A cut-in is therefore
   anticipated *before* the motorcycle is geometrically in-lane.
2. **Dynamic safety re-weighting.** `w_safe` is scaled up when lateral motion is
   detected: `w_safe ← w_safe·(1 + β·|ḋ_moto|)`.

The baseline predicts obstacles with constant lateral position (no intent), so
it reacts only once the motorcycle has already entered the lane.

### 4.3 Vehicle kinematics and feasibility (CLO2)

The ego follows the kinematic single-track (bicycle) model with state
(x, y, θ, v) and inputs (a, δ):

> ẋ = v·cosθ,  ẏ = v·sinθ,  θ̇ = (v/L)·tanδ,  v̇ = a,  with path curvature κ = tanδ / L.

Feasibility of a sampled trajectory therefore requires bounded longitudinal
acceleration |a| ≤ a_max (enforced in the planner) and bounded curvature
|κ| ≤ κ_max = tan(δ_max)/L. The narrow lateral sampling (d ∈ [−0.6, 0.6] m,
single lane) keeps the lateral excursion — and hence the induced curvature —
small, so sampled paths stay within the steering limit. The **kinematic
singularity** of this model is at v → 0, where heading is no longer
controllable through steering (θ̇ ∝ v): in dense stop-and-go the planner must
resolve conflicts longitudinally (braking) rather than by steering, which is
exactly the regime our motorcycle cut-in scenarios exercise. Vehicle parameters
(length 4.5 m, width 1.6 m; motorcycle 2.0 × 0.7 m) follow the pitch calibration
table and the `commonroad-vehicle-models` conventions.

## 5. Implementation and reproducibility (CLO3)

Pure `numpy` / `scipy` / `opencv` + `commonroad-io`; no GPU required. The
planner is a self-contained Werling implementation (not the `commonroad_rp`
package) so every line is inspectable and the drivability-checker C++ build is
avoided. One command regenerates all artifacts: `python src/make_demo.py`. The
live demo centerpiece, `python src/planner_demo.py`, runs in ~3 s.

## 6. Evaluation (CLO1, CLO3, CLO4)

### 6.1 Single showcase (`outputs/planner_metrics.md`)

A motorcycle cut-in on the real GPS reference path, identical for both planners:

| Metric | Baseline | Moto-aware |
|---|---|---|
| Collision | **YES** | none |
| Min clearance | −0.07 m (FAIL 0.3 m) | 2.82 m (pass) |
| TTC exposure (<2 s) | 10% | 0% |
| Peak deceleration | 2.6 m/s² | 1.6 m/s² |
| Time ratio R_T | 1.51 | 1.50 |

The baseline collides; the moto-aware planner avoids it at the **same travel
time** — safety at near-zero efficiency cost.

### 6.2 Scenario suite (`outputs/planner_evaluation.md`)

24 scenarios (3 real GPS road segments × 8 motorcycle behaviours) × 3 planner
variants (baseline; conservative = baseline with large fixed buffers and low
speed; moto-aware). Overall:

| Variant | Collision rate | Clearance pass (≥0.3 m) | Mean R_T | Mean TR1 cost\* |
|---|---|---|---|---|
| baseline | 38% | 50% | 1.12 | 43.8 |
| conservative | 0% | 100% | 1.72 | 23.3 |
| **moto-aware (ours)** | **12%** | **88%** | **1.35** | 59.9 |

\*TR1 = the CommonRoad 2024 competition cost (Huang et al. 2025, §3.2: jerk +
steering-rate + lane-offset + obstacle-distance, weights [0.01, 22, 8, 5]),
averaged only over collision-free runs per their §3.1 (lower = better quality).

By family — the key evidence that the gain is *targeted*:

| Family | baseline (coll / R_T) | moto-aware (coll / R_T) |
|---|---|---|
| no_conflict | 0% / 1.01 | 0% / **1.01** |
| cut_in | 75% / 1.21 | 25% / 1.65 |
| lane_split | 0% (50% pass) / 1.05 | 0% (100% pass) / 1.09 |

**Reading the result.** (i) The improvement is *general*: collisions 38% → 12%
across the whole suite, not one hand-picked scenario. (ii) It is *targeted*: in
no-conflict scenarios moto-aware has identical efficiency to baseline
(R_T 1.01 = 1.01) — it only intervenes on real lateral risk. (iii) It beats
*naive caution*: the conservative planner is fully safe but slow everywhere
(R_T 1.72 vs 1.35).

### 6.3 Alignment with the CommonRoad 2024 competition (Huang et al. 2025)

Our work maps directly onto the most recent CommonRoad Motion Planning
Competition, which strengthens its external validity:

- **Same winning paradigm.** The 2024 winner (TUM-2024) is a *sampling planner
  in the Frenét frame* with lattice quintic trajectories, kinematic-feasibility
  and collision checks, and a 3 s horizon — the same paradigm as our
  `frenet_planner.py`. We did not pick an arbitrary method; we built on the
  competition-winning one.
- **We target the winner's documented weakness.** The report attributes the
  winner's lower trajectory quality to a *"simplified prediction model of other
  agents"* that yields *"overly conservative"* behaviour, and recommends *"more
  accurate prediction models … reducing overly conservative behavior."* Our
  motorcycle-aware prediction (uncertainty-buffer propagation) is exactly that —
  and §6.2 shows it stays non-conservative (R_T = baseline when safe).
- **Standardized metric (TR1).** We score trajectory quality with the
  competition cost TR1. As expected, the conservative planner has the lowest TR1
  (it avoids interaction), while our planner — like the sampling winner relative
  to the optimization runner-up — trades some TR1 to *engage* close motorcycle
  interactions and to succeed on harder scenarios.
- **Traffic-rule mapping.** Our metrics align with the competition's evaluated
  rules: clearance / TTC ≈ **R_G1** (safe distance); R_T ≈ **R_G4** (do not
  impede traffic flow).

## 7. Limitations (stated honestly)

- **Single drive.** The pipeline is built around one recorded drive; the
  five-trip dataset (PR2) is not all in this checkout.
- **Scripted motorcycle behaviour.** Motorcycle motion is a parameter grid, not
  yet calibrated from real two-wheeler logs; calibration (Wk 15–16) is a
  one-line swap in the `Moto` class.
- **Detection not reproducible here.** YOLO detection and real traffic
  segmentation need the source `.MOV`, which is absent from this machine; the
  3-panel replay falls back to committed labeled stills, and segmentation falls
  back to clearly-marked synthetic counts.
- **Residual failures.** Moto-aware still collides in 3/24 (the most aggressive
  cut-ins) — the tuning + calibration target.
- **CARLA 3D not integrated.** CARLA needs an NVIDIA GPU + Linux/Windows;
  unavailable on the team's Apple-Silicon machine. The map → OpenDRIVE half of
  the interface is feasible offline; 3D is future work.

## 8. Feasibility and impact (CLO4)

The pipeline is software-only and runs on a commodity laptop (no GPU), fully
reproducible in ~2.5 min, with near-zero data-collection cost (a smartphone
running Sensor Logger). For Vietnam, ADAS/AV systems must handle
motorcycle-dominant traffic; a planner that anticipates cut-ins directly reduces
collision risk — with clear safety, liability and insurance relevance — and a
simulation-first workflow cuts the cost and risk of on-road testing. Deployment
path: calibrate motorcycle distributions → validate in CARLA 3D → on-vehicle.

## 9. Course-outcome mapping

| CLO | Status | Evidence |
|---|---|---|
| CLO1 (math foundations) | satisfied | coordinate transforms, WGS-84→Mercator, Frenet frame, bicycle kinematics (§4.3) |
| CLO2 (kinematics/dynamics) | satisfied | kinematic single-track model, curvature feasibility, v→0 singularity (§4.3) |
| CLO3 (simulation/implementation) | satisfied | reproducible pipeline (§5), one-command regen |
| CLO4 (safety/efficiency/impact) | satisfied | metrics (§6) + feasibility/impact (§8) |
| CLO7 (teamwork) | satisfied | role split (§10), shared repo + PRs |
| CLO8 (communication) | satisfied | this report + slides + demo (PRESENTATION.md) |

## 10. Contributions and future work

- **Do Minh Phung** — map/scenario, reactive planner + evaluation, repo, report.
- **Vu Thanh Quang** — SUMO/CARLA integration, GoPro 360 capture and analysis.
- **Nguyen Quoc Linh** — motorcycle behaviour calibration, ablation, docs.

**Future work:** calibrated VN-rider distributions; CARLA 3D validation; a
motorcycle-specific car-following model; submission to a CommonRoad competition.

---

*Artifacts:* `src/` (pipeline), `outputs/` (videos, metrics, slides — git-ignored,
regenerate with `make_demo.py`), `PRESENTATION.md` (demo script),
`README.md` / `README2.md` (usage / classmate summary).
