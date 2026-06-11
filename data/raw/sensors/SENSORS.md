# Sensor data — iPhone 12 (Sensor Logger 1.59)

First-person dashcam recording, Hanoi, paired with `data/raw/Car_OCP_3p_2605.MOV`.

| Field           | Value                                  |
|-----------------|----------------------------------------|
| Device          | iPhone 12, iOS 17.5.1                  |
| App             | Sensor Logger 1.59                     |
| Recording start | 2026-05-26 09:10:20 Asia/Ho_Chi_Minh   |
| Duration        | ~225 s (matches the 219.9 s dashcam)   |
| Sample period   | 10 ms target (~100 Hz) for IMU streams |

Every row in every CSV carries:
- `time` — epoch nanoseconds (UTC), monotonic, use this to join streams.
- `seconds_elapsed` — float seconds since recording start. Can be slightly negative for samples captured before the canonical `t=0`.

## Files

| File                              |  Rows |  Hz  | Columns (besides `time`, `seconds_elapsed`)                                                                                            |
|-----------------------------------|------:|-----:|----------------------------------------------------------------------------------------------------------------------------------------|
| `Accelerometer.csv`               | 22504 | ~100 | `x, y, z` — linear acceleration **with gravity removed** (g)                                                                            |
| `AccelerometerUncalibrated.csv`   | 22506 | ~100 | `x, y, z` — raw, **gravity still in** (g)                                                                                              |
| `Gyroscope.csv`                   | 22504 | ~100 | `x, y, z` — angular velocity (rad/s), bias-corrected                                                                                   |
| `GyroscopeUncalibrated.csv`       | 22506 | ~100 | `x, y, z` — angular velocity (rad/s), raw                                                                                              |
| `Gravity.csv`                     | 22504 | ~100 | `x, y, z` — gravity vector (m/s²) in device frame; `√(x²+y²+z²) ≈ 9.81`                                                               |
| `Orientation.csv`                 | 22504 | ~100 | `yaw, pitch, roll` (rad) + quaternion `qw, qx, qy, qz`                                                                                |
| `Magnetometer.csv`                | 22504 | ~100 | `x, y, z` — magnetic field (µT), calibrated                                                                                            |
| `MagnetometerUncalibrated.csv`    | 22471 | ~100 | `x, y, z` — magnetic field (µT), raw (includes hard-iron bias)                                                                         |
| `Compass.csv`                     | 22504 | ~100 | `magneticBearing` (degrees from magnetic north)                                                                                        |
| `Barometer.csv`                   |   211 |   ~1 | `pressure` (hPa), `relativeAltitude` (m, zeroed at recording start)                                                                    |
| `Location.csv`                    |   228 |   ~1 | `latitude, longitude` (WGS-84 °), `altitude, altitudeAboveMeanSeaLevel` (m), `speed` (m/s), `bearing` (° from true north), `horizontalAccuracy, verticalAccuracy, speedAccuracy, bearingAccuracy` |
| `Annotation.csv`                  |     0 |    – | Manual annotations (none recorded)                                                                                                     |
| `Metadata.csv`                    |     1 |    – | Device + recording info (see top of this file)                                                                                         |

## Coordinate frame (device)

When the phone is held in the standard portrait orientation, with the screen
facing the user and the home button at the bottom:

- **+X** points right (out of the right edge of the phone)
- **+Y** points up (out of the top of the phone)
- **+Z** points out of the screen toward the user

This was a windshield-mounted recording, so the device frame is **not** the
car frame. To recover car-frame accelerations (forward / lateral / vertical)
you need to rotate by the static mounting orientation — easiest path is to use
`Gravity.csv` over the first stationary seconds to estimate the down axis,
then resolve forward/lateral from the heading change during the first turn.

## Sign conventions

- **Accelerometer**: in g units (1 g ≈ 9.81 m/s²). `Accelerometer.csv` already
  has gravity removed (so it's near zero when the phone is sitting still);
  `AccelerometerUncalibrated.csv` keeps gravity.
- **Gyroscope**: in rad/s, right-hand-rule positive around each axis.
- **Orientation**: `yaw` measured CCW from north in the device's horizontal
  plane (range ≈ -π..π); `pitch` and `roll` follow Apple's Core Motion
  convention. Quaternion `(qw, qx, qy, qz)` is the more numerically stable
  representation — prefer it for any rotation math.
- **Compass.magneticBearing**: degrees from **magnetic** north (not true).
  Subtract local magnetic declination (~-0.5° for Hanoi in 2026) to get a
  true bearing.
- **Location.bearing**: course over ground from **true** north (degrees,
  0..360); only meaningful when `speed > ~0.3 m/s` — at standstill the GPS
  bearing is unreliable.

## Joining streams

Different sensors have different timestamps, even with the same nominal
`sampleRateMs`. Two patterns work:

1. **Resample to a common grid via `np.interp`** on `seconds_elapsed`
   (used by `cr_gps_drive.py`, `stats_panel.py`, `path_viz/`).
2. **`pandas.merge_asof`** on `time` (nanoseconds) for nearest-neighbour
   joins when you need to keep raw sample points.

The IMU streams (Accel/Gyro/Gravity/Orientation/Mag/Compass) share a sample
clock and align to within a few hundred µs. GPS (`Location`) is ~1 Hz and
needs interpolation against the IMU clock for any tight fusion.

## Quick stats for this recording

- **Trip duration (Location)**: ~227 s, 228 samples
- **Distance**: ~1.49 km
- **Speed**: 0 → 42.7 km/h (peak), ~24 km/h average while moving
- **Region**: Hanoi (lat 20.990°–20.998° N, lon 105.936°–105.946° E)
- **Heading**: predominantly NW (~300°) along OceanPark's main avenue
- **Barometer drift**: relativeAltitude wandered ±0.5 m (sub-floor noise);
  not useful for the absolute height of this short, flat trip but fine for
  detecting bridges/ramps on longer drives.
- **Location.horizontalAccuracy**: ~3–10 m typical; spikes during the
  initial GPS fix in the first few seconds.

## Known sharp edges

- The first one or two GPS samples have `seconds_elapsed < 0` (fix arrived
  before the canonical recording start). Rebase to 0 before joining.
- `Location.bearing` is 0 (not NaN) when the device is stationary. Filter
  by `speed > 0.3 m/s` or by `bearingAccuracy < 30°` before using it.
- Uncalibrated streams (`*Uncalibrated.csv`) are larger because they're not
  filtered for outliers; only use them if you specifically need raw
  sensor output (e.g., for your own bias estimation).
- Quaternion sign is not canonicalised across samples — `qw` can flip sign
  when the orientation crosses a hemisphere boundary. If you difference
  quaternions, normalise the sign first (flip the quaternion when
  `dot(q_prev, q_curr) < 0`).
