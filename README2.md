# README2 — Planner update (read this first)

> For Quang & Linh. This is the short version of what changed and why.
> Original setup/usage is still in [`README.md`](README.md).

## TL;DR (30 seconds)

1. The professor asked to see the **car planner's method improvement** (the thing
   we promised in the proposal, but PR2 only showed perception/replay).
2. So we added a real **reactive planner** + the **motorcycle-aware** version of
   it, and a demo that proves the improvement with numbers.
3. **Result:** baseline planner *crashes* into a cutting-in motorcycle; our
   improved planner *doesn't* — and takes the same travel time. Run one command
   to see it: `python src/planner_demo.py`.

---

## Why we did this

In the proposal (obj. 3) and PR1 (§3.3) we promised to **improve the planner's
cost function for motorcycles**. PR2 quietly pushed that to "later" and only
showed the map + camera + stats replay. The professor noticed and asked: *where
is the planner method improvement?*

This update answers exactly that question.

---

## What the repo had vs. has now

| | Before | After |
|---|---|---|
| YOLO detection, GPS-on-map replay, stats panel | ✅ | ✅ |
| **A planner** | ❌ (not in repo) | ✅ `src/frenet_planner.py` |
| **The motorcycle method improvement** | ❌ | ✅ (the `moto_aware` cost mode) |
| **Proof it works (video + metrics)** | ❌ | ✅ `python src/planner_demo.py` |

---

## The one command

```bash
pip install -r requirements.txt      # adds scipy; everything else you already have
python src/planner_demo.py
```

It produces:
- `outputs/planner_compare.mp4` — **side-by-side video**: baseline (left) vs
  improved (right), driving on our real VinUni / Ocean Park map.
- `outputs/planner_metrics.md` — the **evaluation table** (same metrics as PR2).

No CARLA, no SUMO, no extra downloads needed.

---

## What it shows (the result)

A motorcycle cuts into the ego car's lane (the classic VN "lane-split / cut-in").
Same scenario for both planners — **only the cost function is different.**

| Metric (from PR2) | Baseline | Improved (moto-aware) |
|---|---|---|
| Collision | **YES** 💥 | none ✅ |
| Min clearance (need ≥ 0.30 m) | **−0.07 m (FAIL)** | 2.82 m (pass) |
| Time-to-collision exposure | 10% | 0% |
| Travel-time ratio R_T | 1.51 | 1.50 |

**Plain English:** the improved planner avoids the crash **without driving any
slower** (same R_T). That is the "method improvement" the professor wanted.

**How the improvement works (2 ideas, from PR1 §3.3):**
1. *Uncertainty buffer* — when the motorcycle moves sideways, we make its
   predicted "danger zone" bigger, so the car reacts to the cut-in **early**.
2. *Safety re-weighting* — when sideways motion is detected, the planner cares
   more about safety in that moment.

The baseline has neither, so it only reacts once the motorcycle is already in
front — too late.

---

## New files (if you want to look inside)

| File | What it is |
|---|---|
| `src/frenet_planner.py` | The planner. Frenet-frame sampling (Werling), quintic/quartic trajectories, two cost modes: `baseline` and `moto_aware`. Pure numpy — easy to read. |
| `src/planner_demo.py` | Sets up the cut-in scenario on our real GPS route + map, runs both planners, makes the video and the metrics table. Scenario knobs are all named at the top. |

---

## Honest limits (so we tell the professor the truth)

- This is **our own** lightweight planner that implements the proposal's method,
  not the official `commonroad_rp` package (avoids a painful C++ install). Faithful
  to Werling, and we can explain every line.
- The motorcycle here follows a **scripted** cut-in. Plugging in Linh's
  **calibrated** motorcycle parameters (from the 5-trip data) is the Week 15–16
  step — the code is ready for that swap.
- Scenario numbers in `planner_demo.py` are tuned to make the difference clear;
  they're all at the top of the file and easy to change.

---

## P2 update — 3-panel replay + traffic segmentation (done)

Two more tools, matching PR2:

- `python src/compose_panels.py` → `outputs/replay_3panel.mp4`: the single
  **3-panel replay** from PR2 Fig. 1 (map ∣ camera+detection ∣ stats), with a
  title bar and a **low/high traffic-mode badge**.
- `python src/segment_traffic.py` → splits the trip into **low/high-traffic
  segments** (PR2 Sec. 2.3 rule: mean actors/frame > 4 = high). Outputs a
  segment table, a timeline, and the time-ranges for the selectable demo modes.
- `src/label_obstacles.py` now also writes `detection_counts.csv` (per-frame
  actor counts + nearest range) — the input for segmentation and TTC.

> **Data dependency (important):** the middle camera panel and the *real*
> segmentation need the source dashcam `.MOV` + YOLO, which only run where the
> footage lives (not on a Mac without it). Until then, `compose_panels.py` uses
> the 5 committed labeled stills as a slideshow (so the layout is final and
> verifiable), and `segment_traffic.py --selftest` validates the logic. To
> finish with real data on the footage machine:
> ```bash
> python src/label_obstacles.py      # -> labeled video + detection_counts.csv
> python src/segment_traffic.py      # -> real low/high segments
> python src/compose_panels.py       # -> real 3-panel replay with the camera video
> ```

## What's left (suggested owners)

| Item | Status | Owner |
|---|---|---|
| Planner method improvement demo | ✅ done | Phung |
| 3-panel replay video (PR2 Fig. 1) | ✅ tooling done (real video drops in with footage) | Phung |
| Low/high-traffic split from detection counts | ✅ tooling done (run on footage machine) | Phung/Linh |
| Run detection on the 5 trips → counts + labeled video | ⬜ needs footage machine | Phung/Quang |
| Feed calibrated VN motorcycle params into the planner | ⬜ Wk 15–16 | Linh |
| CARLA 3D demo — **blocked on this Mac** (no NVIDIA GPU); needs a Linux/Win GPU box or cloud GPU. Map→OpenDRIVE half can run offline. | ⬜ if time | Quang |
