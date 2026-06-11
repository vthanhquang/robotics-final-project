"""
Render the real GPS drive onto the VinUni CommonRoad map.

Approach
--------
1. Load Location.csv, project WGS-84 -> EPSG-3857.
2. Shift GPS centroid onto the campus map centroid (~2900 m offset).
3. Render the road network ONCE as a full-extent OpenCV background image.
4. Per frame: crop the 300 m view window, draw car + speed-coded trail + HUD.
   (No per-frame matplotlib -> fast enough for 220 s @ 15 fps in ~1 min.)

Output: cr_gps_drive.mp4  (1280x720, H.264, 15 fps, 220 s)
"""

import csv, math, os
from pathlib import Path

import cv2
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
# VinUni-1_2: full OceanPark map (4797 lanelets, 4.2 x 4.8 km) — contains the
# real GPS track in its native EPSG:3857 coordinates, no shift needed.
XML_PATH = ROOT / "data" / "maps" / "VinUni-1_2-T1.xml"
GPS_PATH = ROOT / "data" / "raw" / "sensors" / "Location.csv"
OUTPUT   = ROOT / "outputs" / "cr_gps_drive.mp4"

# ── Video settings ────────────────────────────────────────────────────────────
WIDTH, HEIGHT    = 1280, 720
FPS              = 15
VIDEO_DURATION   = 219.9          # seconds (matches dashcam)
VIEW_RADIUS_X    = 150.0          # metres half-width of camera window
VIEW_RADIUS_Y    = VIEW_RADIUS_X * HEIGHT / WIDTH   # keep aspect
BG_SCALE         = 3              # background pixels per metre
TRAIL_SECS       = 40             # seconds of trail to show behind car

CAR_LENGTH, CAR_WIDTH = 4.508, 1.610

R_EARTH = 6_378_137.0


# ── Coordinate helpers ────────────────────────────────────────────────────────
def wgs84_to_m(lat, lon):
    x = lon * math.pi * R_EARTH / 180
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360)) * R_EARTH
    return x, y


def speed_to_bgr(pct):
    pct = max(0.0, min(1.0, pct))
    h = int((1.0 - pct) * 120)   # green(120) -> yellow(60) -> red(0)
    hsv = np.uint8([[[h, 230, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def cardinal(deg):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][round(deg / 45) % 8]


# ── 1. Load + project GPS data ────────────────────────────────────────────────
print("Loading GPS data...")
rows = []
with open(GPS_PATH, newline="") as f:
    for r in csv.DictReader(f):
        rows.append((
            float(r["seconds_elapsed"]),
            float(r["latitude"]), float(r["longitude"]),
            float(r["speed"]),    float(r["bearing"]),
        ))
rows.sort(key=lambda r: r[0])
t_src   = np.array([r[0] for r in rows]); t_src -= t_src.min()
lat_src = np.array([r[1] for r in rows])
lon_src = np.array([r[2] for r in rows])
spd_src = np.clip(np.array([r[3] for r in rows]), 0, None)
brg_src = np.array([r[4] for r in rows])

xy_raw = np.array([wgs84_to_m(la, lo) for la, lo in zip(lat_src, lon_src)])

# ── 2. No coordinate shift — the full VinUni map already contains the GPS ────
xy = xy_raw
print(f"  GPS extent: x=[{xy[:,0].min():.0f},{xy[:,0].max():.0f}] "
      f"y=[{xy[:,1].min():.0f},{xy[:,1].max():.0f}]")

# ── 3. Resample to video timeline ─────────────────────────────────────────────
n_frames = int(FPS * VIDEO_DURATION)
t        = np.linspace(0, VIDEO_DURATION, n_frames)
x_pos    = np.interp(t, t_src, xy[:, 0])
y_pos    = np.interp(t, t_src, xy[:, 1])
speed    = np.interp(t, t_src, spd_src)
bx_i     = np.interp(t, t_src, np.cos(np.radians(brg_src)))
by_i     = np.interp(t, t_src, np.sin(np.radians(brg_src)))
bearing  = np.degrees(np.arctan2(by_i, bx_i)) % 360

smax = max(spd_src) * 1.05

# CommonRoad heading: 0 = East, CCW positive — taken straight from GPS bearing
cr_head = np.pi / 2 - np.radians(bearing)

# ── 3b. Load CommonRoad scenario ──────────────────────────────────────────────
print("Loading CommonRoad scenario...")
scenario, _ = CommonRoadFileReader(str(XML_PATH)).open()

# ── 4. Build background extent (bounded by GPS track + padding) ─────────────
PAD   = 350   # metres padding around GPS track (lanelets outside are clipped)
X_MIN = xy[:, 0].min() - PAD
X_MAX = xy[:, 0].max() + PAD
Y_MIN = xy[:, 1].min() - PAD
Y_MAX = xy[:, 1].max() + PAD

BG_W = int((X_MAX - X_MIN) * BG_SCALE)
BG_H = int((Y_MAX - Y_MIN) * BG_SCALE)

def to_bg(x, y):
    return int((x - X_MIN) * BG_SCALE), int((Y_MAX - y) * BG_SCALE)


# Pre-compute trail pixel positions
trail_px = np.array([to_bg(x_pos[i], y_pos[i]) for i in range(n_frames)],
                    dtype=np.int32)

# ── 5. Render road network background (once) ──────────────────────────────────
print("Rendering road network background...")

# background: light off-white
bg = np.full((BG_H, BG_W, 3), (250, 248, 246), dtype=np.uint8)

# Grid lines every 100 m (subtle)
for gx in np.arange(math.ceil(X_MIN / 100) * 100, X_MAX, 100):
    px, _ = to_bg(gx, Y_MIN)
    cv2.line(bg, (px, 0), (px, BG_H), (230, 228, 226), 1)
for gy in np.arange(math.ceil(Y_MIN / 100) * 100, Y_MAX, 100):
    _, py = to_bg(X_MIN, gy)
    cv2.line(bg, (0, py), (BG_W, py), (230, 228, 226), 1)

# Lanelets: filled + bounds
for ll in scenario.lanelet_network.lanelets:
    lv = ll.left_vertices
    rv = ll.right_vertices
    poly = np.vstack([lv, rv[::-1]])
    poly_px = np.array([to_bg(p[0], p[1]) for p in poly], dtype=np.int32)
    cv2.fillPoly(bg, [poly_px], (220, 220, 220))

for ll in scenario.lanelet_network.lanelets:
    lv_px = np.array([to_bg(p[0], p[1]) for p in ll.left_vertices],  dtype=np.int32)
    rv_px = np.array([to_bg(p[0], p[1]) for p in ll.right_vertices], dtype=np.int32)
    cv2.polylines(bg, [lv_px], False, (90, 90, 90), max(1, BG_SCALE-1), cv2.LINE_AA)
    cv2.polylines(bg, [rv_px], False, (90, 90, 90), max(1, BG_SCALE-1), cv2.LINE_AA)

# Full GPS path (faint blue)
cv2.polylines(bg, [trail_px], False, (200, 180, 140), 2, cv2.LINE_AA)

print(f"  Background: {BG_W}x{BG_H} px  ({(X_MAX-X_MIN):.0f}x{(Y_MAX-Y_MIN):.0f} m)")

# ── 6. View-window helpers ─────────────────────────────────────────────────────
half_vx  = int(VIEW_RADIUS_X * BG_SCALE)    # bg pixels
half_vy  = int(VIEW_RADIUS_Y * BG_SCALE)
sf_x     = WIDTH  / (2 * half_vx)
sf_y     = HEIGHT / (2 * half_vy)

def bg_to_view(bx_, by_, cx_bg, cy_bg):
    vx = int((bx_ - (cx_bg - half_vx)) * sf_x)
    vy = int((by_ - (cy_bg - half_vy)) * sf_y)
    return vx, vy


def draw_car(frame, cx_v, cy_v, heading_rad, length_m, width_m):
    l_px = int(length_m * BG_SCALE * sf_x)
    w_px = int(width_m  * BG_SCALE * sf_y)
    angle_cv = math.degrees(heading_rad)   # cv2 boxPoints uses degrees CW
    box = cv2.boxPoints(((float(cx_v), float(cy_v)),
                         (float(l_px), float(w_px)),
                         float(-angle_cv)))
    box = box.astype(np.int32)
    cv2.fillPoly(frame, [box], (60, 76, 231))     # red #e74c3c in BGR
    cv2.polylines(frame, [box], True, (40, 50, 60), 2, cv2.LINE_AA)


# ── 7. Render video ────────────────────────────────────────────────────────────
print(f"Rendering {n_frames} frames ({VIDEO_DURATION:.1f}s @ {FPS} fps)...")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(str(OUTPUT), fourcc, float(FPS), (WIDTH, HEIGHT))
trail_len = int(FPS * TRAIL_SECS)

for i in range(n_frames):
    cx_bg, cy_bg = trail_px[i]

    # Clamp crop to background bounds
    bx0 = max(cx_bg - half_vx, 0); bx1 = min(cx_bg + half_vx, BG_W)
    by0 = max(cy_bg - half_vy, 0); by1 = min(cy_bg + half_vy, BG_H)
    crop = bg[by0:by1, bx0:bx1]

    # Pad if near edge
    pad_l = max(0, half_vx - cx_bg)
    pad_r = max(0, (cx_bg + half_vx) - BG_W)
    pad_t = max(0, half_vy - cy_bg)
    pad_b = max(0, (cy_bg + half_vy) - BG_H)
    if any([pad_l, pad_r, pad_t, pad_b]):
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(250, 248, 246))

    view = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Speed-coded trail
    t_start = max(0, i - trail_len)
    for j in range(t_start + 1, i + 1):
        p0 = bg_to_view(trail_px[j-1, 0], trail_px[j-1, 1], cx_bg, cy_bg)
        p1 = bg_to_view(trail_px[j,   0], trail_px[j,   1], cx_bg, cy_bg)
        col = speed_to_bgr(speed[j] / smax)
        cv2.line(view, p0, p1, col, 3, cv2.LINE_AA)

    # Car
    cx_v = WIDTH  // 2
    cy_v = HEIGHT // 2
    draw_car(view, cx_v, cy_v, cr_head[i], CAR_LENGTH, CAR_WIDTH)

    # ── HUD ──────────────────────────────────────────────────────────────────
    speed_kmh = speed[i] * 3.6
    progress  = i / n_frames * 100

    # semi-transparent panel
    ov = view.copy()
    cv2.rectangle(ov, (10, 8), (450, 100), (255, 255, 255), -1)
    cv2.addWeighted(ov, 0.55, view, 0.45, 0, view)
    cv2.rectangle(view, (10, 8), (450, 100), (200, 200, 200), 1)

    cv2.putText(view, "VinUni  -  Real GPS Drive",
                (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (26, 26, 26), 2, cv2.LINE_AA)
    cv2.putText(view,
                f"Spd: {speed_kmh:.0f} km/h   Hdg: {bearing[i]:.0f}deg ({cardinal(bearing[i])})   "
                f"t: {t[i]:.1f}s",
                (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)

    # progress bar
    bx0h, by0h, bw, bh = 18, 76, 300, 12
    cv2.rectangle(view, (bx0h, by0h), (bx0h + bw, by0h + bh), (210, 210, 210), -1)
    cv2.rectangle(view, (bx0h, by0h), (bx0h + int(bw * progress / 100), by0h + bh),
                  (185, 128, 41), -1)
    cv2.rectangle(view, (bx0h, by0h), (bx0h + bw, by0h + bh), (170, 170, 170), 1)

    out.write(view)
    if (i + 1) % (FPS * 30) == 0:
        print(f"  {i+1}/{n_frames}  ({t[i]:.0f}s)")

out.release()
print(f"\nDone: {OUTPUT}")
