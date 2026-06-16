# Slide content — APEX planner (paste-ready)

## Presentation links / files
- **Canva (present/view):** https://www.canva.com/d/mb88rtVRfhOFmh3
- **Canva (edit):** https://www.canva.com/d/RzKxfcome1U7T9o
- _(8-slide v4: cover, problem, **APEX architecture — inheritance**, real-data (PR2)+screenshot, **APEX core**, results, **learning-augmented MPC**, demo)_
- Architecture from `apex_architecture.html` → `slides/apex_arch_{inherit,core,learn}.png` (build: `python src/render_html_sections.py`)
- **PPTX (VinUni template):** `slides/Vietnam-MixedTrafficSim_APEX.pptx` (build: `python src/make_pptx.py`)
- **PDF deck:** `slides/SLIDES_APEX.pdf`  ·  **Demo reel:** `outputs/apex_reel.mp4`  ·  **Slide PNGs:** `outputs/slide_*.png`


Source numbers: Quang's 52-scenario benchmark (`planner-scenario-suite` branch /
`slide_planning.md`). Rendered slides: `outputs/slide_{1,2,3}_*.png`
(`python src/make_slides.py`). Fallback reel: `outputs/apex_reel.mp4`
(`python src/apex_reel.py`).

## % improvements (the headline numbers)

| APEX vs … | Collisions | <0.3 m fails | Efficiency (R_T) |
|---|---|---|---|
| baseline Frenet | 28 → 0 = **−100%** | 30 → 0 = **−100%** | +12% time (baseline crashes 28×) |
| moto-aware (our previous) | 13 → 0 = **−100%** | 14 → 0 = **−100%** | +7% time for 0 crashes |
| IDM (Treiber 2000) | 5 → 0 = **−100%** | 7 → 0 = **−100%** | **9% faster** → strictly better |
| ORCA/VO (van den Berg 2011) | 1 → 0 = **−100%** | 1 → 0 = **−100%** | +7% time for the last crash |
| conservative (other crash-free) | 0 = 0 | 0 = 0 | **23% faster** (1.43 vs 1.86) |

**Three numbers to say out loud:**
1. **0 collisions / 0 clearance-fails across all 52 scenarios** — −100% vs baseline (28) and vs our previous moto-aware (13).
2. **23% faster than the only other crash-free planner** (conservative), with a realistic 2.16 m clearance instead of a 13.96 m standoff.
3. **Beats both published baselines** (IDM, ORCA/VO); junction **crossings solved 0/4**.

## Slide 1 — Problem
- AV planners assume Western lane-abiding traffic; Vietnam is motorcycle-dominated (lane-split, sub-meter cut-ins, crossings).
- Goal: a planner that *anticipates* motorcycle conflicts on the real VinUni road.

## Slide 2 — Method: APEX predictive risk-aware planner
*Fuses car-following + reactive avoidance + sampling; fixes each one's flaw.*
1. Predict every motorcycle over a 4 s horizon + reachability envelope → anticipates a cut-in before it starts.
2. Hard footprint-clearance margin to all motos over the whole horizon → **collision-free by construction**.
3. Junction-yield speed-cap for roadside / crossing motos → solves crossings (0/4).
4. Multi-horizon trajectories → brake hard when needed, full speed when clear (no over-braking).
5. Speed-maximizing objective under the safety constraint → most efficient among safe planners.

## Slide 3 — Results: 52 scenarios × 6 planners

| Planner | Collisions | <0.3 m fails | Min-clr | R_T |
|---|---|---|---|---|
| IDM (Treiber 2000) | 5/52 | 7/52 | 5.73 | 1.57 |
| ORCA/VO (van den Berg 2011) | 1/52 | 1/52 | 2.02 | 1.34 |
| baseline Frenet | 28/52 | 30/52 | 1.24 | 1.27 |
| conservative | 0/52 | 0/52 | 13.96 | 1.86 |
| moto-aware (prev. ours) | 13/52 | 14/52 | 2.31 | 1.33 |
| **APEX (predictive, ours)** | **0/52** | **0/52** | **2.16** | **1.43** |

> Only conservative and APEX are crash-free — and APEX is the efficient one.

## Live demo (6 min) — the 6 scenario videos
Each is a 4-panel comparison (baseline | ORCA/VO | moto-aware | APEX): others collide, APEX stays safe.

Re-render the 4-panel comparison videos with `python src/scenario_videos_apex.py`
(baseline | ORCA/VO | moto-aware | APEX; clean "APEX (predictive, ours)" label):

| Order | Video | Family |
|---|---|---|
| 1 | `Phung_cut_in_C07.mp4` | Cut-in |
| 2 | `Phung_crossing_X01.mp4` | Junction crossing |
| 3 | `Phung_shoulder_merge_H01.mp4` | Shoulder merge |
| 4 | `Phung_multi_lane_W01.mp4` | Multi-lane weave |
| 5 | `Phung_two_moto_T2-05.mp4` | Two motorcycles |
| 6 | `Phung_three_moto_T3-03.mp4` | Three motorcycles |

Lead with **crossing** and **three-moto** (strongest contrast). Fallback: play `outputs/apex_reel.mp4` (~85 s) and narrate.
