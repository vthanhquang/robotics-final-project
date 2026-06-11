# path_viz — GPS path visualization (isolated)

Standalone driving-path video rendered from `data/raw/sensors/Location.csv`,
timeline-synced to the dashcam footage.

## Files
- `make_path_video.py` — renderer (pure NumPy + OpenCV, no plotting deps).
- `path_video.mp4`     — output, 960×720 @ 30 fps, ~220 s (matches dashcam).
- `preview_*.jpg`      — sanity stills at 1 / 25 / 50 / 75 / 99 %.

## What it shows
- Full route in grey; traveled portion drawn in a speed-coded colour ramp
  (blue → green → yellow → red, 0 → 50 km/h).
- Rotating top-down car icon at the current position — pentagon nose-tip
  always points in the direction of motion (interpolated GPS bearing).
- START (green) / END (red) markers, north arrow.
- Translucent HUD: elapsed time, speed (km/h), heading (° + cardinal),
  cumulative distance, current lat/lon.
- Semicircular speedometer gauge in the bottom-right with needle + digital
  readout, and a speed colour-ramp legend bar.

## Method
1. Read `Location.csv` (~1 Hz GPS, 229 rows, 227 s span).
2. Rebase `seconds_elapsed` to 0; clip negative `speed` values.
3. Resample to 30 Hz over the video duration via `np.interp`.
   Bearing is interpolated on `(cos θ, sin θ)` to handle 0/360 wrap.
4. Project lon/lat with `lon × cos(mean_lat)` (equirectangular, preserves
   local aspect at this latitude).
5. Pre-render a static base (title, full grey track, START/END, N arrow,
   legend bar).
6. Maintain a `trail` canvas: per frame, draw the new segment in the colour
   for the current speed, then composite onto a copy of the base.
7. Rotate the BGRA car sprite via `cv2.warpAffine` by the current heading,
   alpha-blend it at the projected position.
8. Draw the speedometer gauge and HUD panel on top.

## Trip summary (from this run)
- Distance: **1.49 km**
- Duration: ~220 s
- Peak speed: ~43 km/h
- Region: Hanoi (lat 20.990–20.998, lon 105.936–105.946)

## To run
```bash
python path_viz/make_path_video.py
```

## To remove this version
Delete the whole `path_viz/` folder — nothing else depends on it.
