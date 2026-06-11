# Sensor data overview

Collected with an **iPhone 12** running the **Sensor Logger** app (v1.59), held
in the windshield for a ~220 s drive in Hanoi. Lives in `raw/sensors/` as one
CSV per stream; each row has `time` (epoch ns) and `seconds_elapsed`.

## What's in it

| Sensor                | What it measures                            | Typical use                                                |
|-----------------------|---------------------------------------------|------------------------------------------------------------|
| Accelerometer         | Linear acceleration (g), gravity removed    | Detect braking, accel events, road bumps                   |
| Gyroscope             | Angular velocity (rad/s)                    | Turn rate, lean / roll dynamics                            |
| Gravity               | Gravity vector in device frame              | Recover the "down" axis, calibrate phone-to-car mounting   |
| Orientation           | Yaw / pitch / roll + quaternion             | Vehicle heading & attitude, dead reckoning                 |
| Magnetometer          | Magnetic field (µT)                         | Heading reference, compass calibration                     |
| Compass               | Magnetic bearing (°)                        | Quick heading readout vs. north                            |
| Barometer             | Air pressure + relative altitude (~1 Hz)    | Bridge / ramp detection, elevation profile                 |
| Location (GPS)        | Lat / lon / altitude / speed / bearing (~1 Hz) | Trajectory, speed profile, map-matching to road network |
| *Uncalibrated variants of accel / gyro / mag | Same as above, raw (bias-uncorrected) | Custom bias estimation, sensor-fusion research          |
| Annotation            | Manual time-stamped notes (empty here)      | Marking events (intersections, near-misses)                |
| Metadata              | Device + recording info                     | Reproducibility                                            |

## Sampling

- **IMU streams** (accel, gyro, gravity, orientation, mag, compass): ~100 Hz,
  ~22,500 rows each.
- **Barometer**: ~1 Hz, 211 rows.
- **GPS**: ~1 Hz, 228 rows.

Join IMU streams on `time` directly (they share a clock). For tight fusion
with GPS or barometer, resample those up to the IMU rate via interpolation.
