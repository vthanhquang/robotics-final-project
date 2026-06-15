# Final demo — presentation script (15-minute slot)

**Strict 15 min:** 4 min talk · 6 min live demo · 5 min Q&A.
Slides: `outputs/slide_{1,2,3}_*.png` (run `python src/make_slides.py`).
Reel fallback: `outputs/demo_reel.mp4` (run `python src/demo_reel.py`).

---

## Part 1 — Presentation (4 min, ~80 s/slide)

### Slide 1 — The Problem  (`slide_1_problem.png`)
- AV motion planners assume Western traffic (marked lanes, rule-abiding drivers).
- In Vietnam >90% of vehicles are motorcycles: lane-splitting, sub-meter cut-ins, ignoring signals.
- Standard planners react too late → unsafe.
- **Our goal:** a planner that *anticipates* motorcycle cut-ins, on the real VinUni road.

### Slide 2 — The Method  (`slide_2_method.png`)
- Frenet-frame sampling reactive planner (Werling): quintic/quartic candidates, scored on jerk + speed + lateral + safety.
- **Improvement (proposal obj. 3 / PR1 §3.3):**
  1. **Uncertainty-buffer propagation** — the motorcycle's predicted footprint inflates with its lateral velocity → the cut-in is anticipated early.
  2. **Dynamic safety re-weighting** — safety cost rises when lateral motion is detected.
- Runs on the real VinUni CommonRoad map + recorded GPS.

### Slide 3 — Results  (`slide_3_results.png`)
- 24 scenarios × 3 planners. Collisions **38% → 12%**, clearance-pass **50% → 88%**.
- **Targeted**, not just slower: same efficiency as baseline when safe (R_T 1.01 = 1.01).
- Safer *and* faster than naive caution (1.35 vs 1.72). → *"Let me show it live."*

---

## Part 2 — Live demonstration (6 min)

| Time | Action | Type |
|---|---|---|
| 0:00–1:30 | Play `outputs/replay_3panel.mp4` — real GPS on the map ∣ YOLO detection ∣ drive stats (PR2 data pipeline) | pre-rendered |
| 1:30–3:30 | **Run live:** `python src/planner_demo.py` (~3 s) → open `outputs/planner_compare.mp4`. Left baseline collides; right moto-aware keeps 2.82 m clearance at the same travel time | **LIVE** |
| 3:30–5:30 | Open `outputs/planner_evaluation.md` (or slide 3) — 24-scenario table; emphasise: general, targeted, beats conservative; honest about the residual 12% | table |
| 5:30–6:00 | VinUni map context + one-line close | buffer |

**If anything hiccups:** play `outputs/demo_reel.mp4` and narrate.

---

## Part 3 — Q&A (5 min) — prepared answers

- **"Is the motorcycle behaviour real data?"** Real road geometry + real GPS; motorcycle behaviour is scripted via a parameter grid. Calibrating it from the 5-trip + motorcycle logging is the next step (Linh, Wk 15–16) — a one-line swap in the `Moto` class.
- **"Why does moto-aware still collide 12%?"** The most aggressive cut-ins. A conservative planner gets 0% but needs +37% travel time; our contribution is the safety/efficiency trade-off. Tuning + calibration is the path to closing the gap.
- **"What exactly is the baseline?"** A standard Werling Frenet reactive planner with no modelling of obstacle lateral intent.
- **"How is this different from CommonRoad's planner?"** A cost-function modification: uncertainty-buffer propagation + dynamic safety re-weighting for two-wheeler lateral intent.
- **"Why no CARLA 3D?"** CARLA needs an NVIDIA GPU + Linux/Windows; our method contribution lives in 2D planning/evaluation. The map→OpenDRIVE conversion (the offline half of the interface) is feasible; 3D visualization is future work.
- **"Are the metrics valid?"** PR2 metrics — collision, clearance (0.3 m), TTC (<2 s), AEB, R_T — over 24 scenarios on 3 real road segments.

---

## Pre-demo checklist

**P0 (must):**
- [ ] On the demo laptop: clean clone → `pip install -r requirements.txt` → `python src/make_demo.py` succeeds.
- [ ] `python src/make_slides.py` and `python src/demo_reel.py`.
- [ ] Test fullscreen playback at the projector resolution.
- [ ] Screen-record the whole demo as a fallback.
- [ ] Rehearse to fit 4 / 6 / 5 exactly.

**P1 (recommended):**
- [ ] Run `python src/label_obstacles.py` on the machine with the source `.MOV` → real camera panel + `detection_counts.csv`, then re-run `make_demo.py` for the *real* 3-panel replay and segmentation.

## One-command prep
```bash
python src/make_demo.py     # regenerate every artifact (~2.5 min)
python src/make_slides.py   # slides
python src/demo_reel.py     # fallback reel
```
