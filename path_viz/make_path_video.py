"""
Render a driving-path visualization from the iPhone Sensor Logger GPS track.
Output is a standalone mp4 whose duration / fps match the source dashcam video,
so it can be played side-by-side with data/outputs/Car_OCP_labeled.mp4.

This file is fully isolated — delete the path_viz/ folder to remove this version.
"""

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
LOCATION = ROOT / "data" / "raw" / "sensors" / "Location.csv"
OUTPUT   = HERE / "path_video.mp4"

# ── Video output config ─────────────────────────────────────────────────────
VIDEO_FPS      = 30.0
VIDEO_DURATION = 219.9
FRAME_W        = 960
FRAME_H        = 720
PAD_X, PAD_TOP, PAD_BOT = 40, 150, 40


# ── Geometry helpers ────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlmb   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def cardinal(deg):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(deg / 45) % 8]


# ── Colour helpers ──────────────────────────────────────────────────────────
def speed_to_bgr(pct):
    """pct in [0,1] → BGR colour ramping blue → green → yellow → red."""
    pct = max(0.0, min(1.0, pct))
    h_deg = (1.0 - pct) * 240.0          # 240° (blue) → 0° (red)
    hsv = np.uint8([[[int(h_deg / 2), 230, 235]]])   # OpenCV H ∈ [0,179]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ── Sprite (top-down car, nose-up) ──────────────────────────────────────────
def make_car_sprite():
    """Top-down car sprite. Pentagon body, pointed nose at the TOP (y=0)."""
    H, W = 64, 38
    s = np.zeros((H, W, 4), dtype=np.uint8)

    body = np.array([
        [W // 2, 3],         # nose tip
        [W - 5, 16],         # right shoulder
        [W - 5, H - 6],      # right rear corner
        [5, H - 6],          # left rear corner
        [5, 16],             # left shoulder
    ], dtype=np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(s, [body], (40, 40, 220, 255), cv2.LINE_AA)
    cv2.polylines(s, [body], True, (255, 255, 255, 255), 2, cv2.LINE_AA)

    # windshield (front)
    ws = np.array([
        [W // 2, 12],
        [W - 8, 20],
        [W - 8, 28],
        [8, 28],
        [8, 20],
    ], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(s, [ws], (220, 205, 175, 255), cv2.LINE_AA)

    # rear window
    cv2.rectangle(s, (9, H - 22), (W - 9, H - 12), (220, 205, 175, 255), -1)

    # headlights (bright cyan) at front shoulders
    cv2.circle(s, (10, 17), 3, (210, 245, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(s, (W - 11, 17), 3, (210, 245, 255, 255), -1, cv2.LINE_AA)

    # tail lights (red) at rear
    cv2.rectangle(s, (7, H - 10),  (13, H - 7),     (30, 30, 240, 255), -1)
    cv2.rectangle(s, (W - 14, H - 10), (W - 8, H - 7), (30, 30, 240, 255), -1)

    return s


def rotate_bgra(sprite, angle_deg):
    """Rotate a BGRA sprite. angle_deg is screen-clockwise positive."""
    H, W = sprite.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), -angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(H * sin + W * cos) + 2
    nH = int(H * cos + W * sin) + 2
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    return cv2.warpAffine(
        sprite, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def paste_bgra(frame, sprite, cx, cy):
    H, W = sprite.shape[:2]
    x0, y0 = cx - W // 2, cy - H // 2
    x1, y1 = x0 + W, y0 + H
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sx0, sy0 = fx0 - x0, fy0 - y0
    sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)
    crop  = sprite[sy0:sy1, sx0:sx1]
    alpha = crop[..., 3:4].astype(np.float32) / 255.0
    bgr   = crop[..., :3].astype(np.float32)
    bg    = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (alpha * bgr + (1 - alpha) * bg).astype(np.uint8)


# ── Speedometer (bottom-arc gauge) ──────────────────────────────────────────
def draw_speedometer(frame, speed_kmh, max_kmh, cx, cy, r=58):
    pct = max(0.0, min(speed_kmh / max_kmh, 1.0))

    cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, 180, (55, 55, 55), 3, cv2.LINE_AA)
    fill_start = int(180 - pct * 180)
    if pct > 0.005:
        cv2.ellipse(frame, (cx, cy), (r, r), 0, fill_start, 180,
                    speed_to_bgr(pct), 5, cv2.LINE_AA)

    for k in range(0, int(max_kmh) + 1, 10):
        a = math.radians(180 - (k / max_kmh) * 180)
        x1 = int(cx + (r - 7) * math.cos(a)); y1 = int(cy + (r - 7) * math.sin(a))
        x2 = int(cx + r * math.cos(a));       y2 = int(cy + r * math.sin(a))
        cv2.line(frame, (x1, y1), (x2, y2), (140, 140, 140), 1, cv2.LINE_AA)

    needle_a = math.radians(180 - pct * 180)
    nx = int(cx + (r - 12) * math.cos(needle_a))
    ny = int(cy + (r - 12) * math.sin(needle_a))
    cv2.line(frame, (cx, cy), (nx, ny), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, (220, 220, 220), -1)

    txt = f"{speed_kmh:.0f}"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
    cv2.putText(frame, txt, (cx - tw // 2, cy + r - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "km/h", (cx - 22, cy + r + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def translucent_panel(frame, x, y, w, h, alpha=0.45):
    sub  = frame[y:y+h, x:x+w].copy()
    rect = np.zeros_like(sub)
    cv2.rectangle(rect, (0, 0), (w, h), (0, 0, 0), -1)
    blended = cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0)
    frame[y:y+h, x:x+w] = blended
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1, cv2.LINE_AA)


def load_location(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((
                float(r["seconds_elapsed"]),
                float(r["latitude"]),
                float(r["longitude"]),
                float(r["speed"]),
                float(r["bearing"]),
            ))
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    rows    = load_location(LOCATION)
    t_src   = np.array([r[0] for r in rows]); t_src -= t_src.min()
    lat_src = np.array([r[1] for r in rows])
    lon_src = np.array([r[2] for r in rows])
    spd_src = np.clip(np.array([r[3] for r in rows]), 0, None)
    brg_src = np.array([r[4] for r in rows])

    n_frames = int(VIDEO_FPS * VIDEO_DURATION)
    t        = np.linspace(0, VIDEO_DURATION, n_frames)
    lat      = np.interp(t, t_src, lat_src)
    lon      = np.interp(t, t_src, lon_src)
    speed    = np.interp(t, t_src, spd_src)
    bx       = np.interp(t, t_src, np.cos(np.radians(brg_src)))
    by       = np.interp(t, t_src, np.sin(np.radians(brg_src)))
    bearing  = np.degrees(np.arctan2(by, bx)) % 360

    seg = np.array([
        haversine_m(lat[i-1], lon[i-1], lat[i], lon[i]) for i in range(1, n_frames)
    ])
    cum_dist = np.concatenate(([0.0], np.cumsum(seg)))
    total_km = cum_dist[-1] / 1000

    # ── Projection ──────────────────────────────────────────────────────────
    mean_lat   = (lat.min() + lat.max()) / 2
    lon_factor = math.cos(math.radians(mean_lat))
    x_src = lon * lon_factor
    y_src = lat

    pad = 0.08
    x_min, x_max = x_src.min(), x_src.max()
    y_min, y_max = y_src.min(), y_src.max()
    x_min -= (x_max - x_min) * pad; x_max += (x_max - x_min) * pad
    y_min -= (y_max - y_min) * pad; y_max += (y_max - y_min) * pad

    W_IN = FRAME_W - 2 * PAD_X
    H_IN = FRAME_H - PAD_TOP - PAD_BOT
    s    = min(W_IN / (x_max - x_min), H_IN / (y_max - y_min))
    ox   = PAD_X  + (W_IN - s * (x_max - x_min)) / 2
    oy   = PAD_TOP + (H_IN - s * (y_max - y_min)) / 2

    def to_px(xv, yv):
        return int(ox + (xv - x_min) * s), int(oy + (y_max - yv) * s)

    path_px = np.array([to_px(x_src[i], y_src[i]) for i in range(n_frames)],
                       dtype=np.int32)

    # ── Per-segment colour (by instantaneous speed) ─────────────────────────
    smax_kmh = max(50, math.ceil(speed.max() * 3.6 / 10) * 10)   # round up to 10
    seg_colors = [speed_to_bgr(speed[i] * 3.6 / smax_kmh) for i in range(n_frames)]

    # ── Static base ─────────────────────────────────────────────────────────
    base = np.full((FRAME_H, FRAME_W, 3), 20, dtype=np.uint8)
    cv2.putText(base, "Driving Path  -  GPS Sensor Track  (iPhone 12, Hanoi)",
                (PAD_X, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(base, "synced to Car_OCP dashcam timeline   |   trail colour = speed",
                (PAD_X, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (170, 170, 170), 1, cv2.LINE_AA)

    cv2.polylines(base, [path_px], False, (75, 75, 75), 1, cv2.LINE_AA)

    sxp, syp = path_px[0]
    cv2.circle(base, (sxp, syp), 8, (0, 200, 0), -1, cv2.LINE_AA)
    cv2.putText(base, "START", (sxp + 12, syp + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    exp, eyp = path_px[-1]
    cv2.circle(base, (exp, eyp), 8, (0, 80, 220), -1, cv2.LINE_AA)
    cv2.putText(base, "END", (exp + 12, eyp + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 110, 240), 1, cv2.LINE_AA)

    nx, ny = FRAME_W - 55, 110
    cv2.arrowedLine(base, (nx, ny + 22), (nx, ny - 18),
                    (240, 240, 240), 2, tipLength=0.4, line_type=cv2.LINE_AA)
    cv2.putText(base, "N", (nx - 6, ny + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    # speed-legend strip (small gradient bar)
    legend_x, legend_y, legend_w, legend_h = FRAME_W - 215, FRAME_H - 28, 180, 10
    for i in range(legend_w):
        c = speed_to_bgr(i / (legend_w - 1))
        cv2.line(base, (legend_x + i, legend_y),
                 (legend_x + i, legend_y + legend_h), c, 1)
    cv2.rectangle(base, (legend_x, legend_y),
                  (legend_x + legend_w, legend_y + legend_h),
                  (200, 200, 200), 1)
    cv2.putText(base, "0", (legend_x - 12, legend_y + legend_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(base, f"{smax_kmh:.0f} km/h",
                (legend_x + legend_w + 6, legend_y + legend_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Render ──────────────────────────────────────────────────────────────
    car_sprite = make_car_sprite()
    trail      = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    trail_mask = np.zeros((FRAME_H, FRAME_W),    dtype=np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(OUTPUT), fourcc, VIDEO_FPS, (FRAME_W, FRAME_H))
    if not out.isOpened():
        sys.exit("Could not open VideoWriter")

    print(f"Rendering {n_frames} frames ({VIDEO_DURATION:.1f}s @ {VIDEO_FPS} fps)")
    print(f"Speed range: 0 .. {smax_kmh:.0f} km/h     Total: {total_km:.2f} km")

    for i in range(n_frames):
        if i > 0:
            p0, p1 = tuple(path_px[i - 1]), tuple(path_px[i])
            cv2.line(trail,      p0, p1, seg_colors[i], 3, cv2.LINE_AA)
            cv2.line(trail_mask, p0, p1, 255,            3, cv2.LINE_AA)

        frame = base.copy()
        m = trail_mask > 0
        frame[m] = trail[m]

        cx, cy = int(path_px[i, 0]), int(path_px[i, 1])
        rotated = rotate_bgra(car_sprite, bearing[i])
        paste_bgra(frame, rotated, cx, cy)

        # ── HUD panel + text ────────────────────────────────────────────────
        translucent_panel(frame, PAD_X - 10, 78, 290, 138)
        speed_kmh = speed[i] * 3.6
        hud = [
            f"t   : {t[i]:6.1f} / {VIDEO_DURATION:.1f} s",
            f"spd : {speed_kmh:5.1f} km/h",
            f"hdg : {bearing[i]:5.1f} deg  ({cardinal(bearing[i])})",
            f"dst : {cum_dist[i]/1000:5.2f} / {total_km:.2f} km",
            f"lat : {lat[i]:.6f}",
            f"lon : {lon[i]:.6f}",
        ]
        for k, line in enumerate(hud):
            cv2.putText(frame, line, (PAD_X, 100 + k * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (235, 235, 235), 1, cv2.LINE_AA)

        # ── Speedometer (bottom-right) ──────────────────────────────────────
        draw_speedometer(frame, speed_kmh, smax_kmh,
                         cx=FRAME_W - 95, cy=FRAME_H - 110, r=58)

        out.write(frame)
        if i % 600 == 0 and i > 0:
            print(f"  {i}/{n_frames} frames")

    out.release()
    print(f"\nDone. Output: {OUTPUT}")


if __name__ == "__main__":
    main()
