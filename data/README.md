# Data

```
data/
├── maps/                       CommonRoad scenarios of the VinUni / Ocean Park road network
│   ├── VinUni-1_1-T1.xml       small extent  (used by render_map_for_marking.py)
│   └── VinUni-1_2-T1.xml       full OceanPark map, 4797 lanelets (used by cr_gps_drive.py)
├── raw/
│   ├── Car_OCP_3p_2605.MOV     dashcam footage  ← NOT in git (see below)
│   └── sensors/                iPhone Sensor Logger CSV exports (+ SENSORS.md spec)
└── outputs/                    sample stills committed for reference; rendered .mp4 are git-ignored
```

## Sensors

`raw/sensors/` holds one CSV per stream from an iPhone 12 (Sensor Logger 1.59),
recorded 2026-05-26 in Hanoi and time-aligned to the dashcam clip. Each row
carries an epoch-nanosecond `time` and a `seconds_elapsed` column so streams can
be joined. See [`raw/sensors/SENSORS.md`](raw/sensors/SENSORS.md) for the full
per-file schema, coordinate frames, and sign conventions, and
[`SENSORS.md`](SENSORS.md) for a higher-level overview.

## Large media (not version-controlled)

The raw dashcam clip and rendered videos are excluded from git (see the repo
`.gitignore`) to keep the clone small. To reproduce the full pipeline, place the
source footage at:

```
data/raw/Car_OCP_3p_2605.MOV      # 848×464 @ 30 fps, H.264, ~220 s
```

Any script that needs it reads from that path. Model weights
(`models/*.pt`) are likewise excluded — `ultralytics` downloads them
automatically on first run.
