"""
Render a 400x450 driving-stats panel synced to Location.csv.

Layout
------
  ┌──────────────────────────┐  TITLE BAR + clock
  │   semicircular speedo    │  current speed + km/h
  │     ──────────────       │
  │   speed-over-trip chart  │  full trip with playhead
  │     ──────────────       │
  │   turn-rate L/R bar      │  derived from bearing
  │     ──────────────       │
  │   heading / dist / max   │  footer
  └──────────────────────────┘
"""

import csv, math
from pathlib import Path

import cv2
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
GPS_PATH = ROOT / "data" / "raw" / "sensors" / "Location.csv"
OUTPUT   = ROOT / "outputs" / "stats_panel.mp4"

WIDTH, HEIGHT = 400, 450
FPS           = 30
DURATION      = 219.9
R_EARTH       = 6_378_137.0

# Palette
BG   = (252, 250, 248)
INK  = (40, 40, 40)
SUB  = (110, 110, 110)
LINE = (210, 210, 210)
ACC  = (185, 128, 41)   # blue/orange BGR
LEFT_C  = (60, 170, 60)
RIGHT_C = (60, 80, 220)


def wgs84_to_m(lat, lon):
    x = lon * math.pi * R_EARTH / 180
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360)) * R_EARTH
    return x, y


def speed_to_bgr(pct):
    pct = max(0.0, min(1.0, pct))
    h = int((1.0 - pct) * 120)
    hsv = np.uint8([[[h, 230, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def cardinal(deg):
    return ["N","NE","E","SE","S","SW","W","NW"][round(deg / 45) % 8]


# ── Load GPS ─────────────────────────────────────────────────────────────────
print("Loading GPS...")
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

# Distance from raw lat/lon
xy_src = np.array([wgs84_to_m(la, lo) for la, lo in zip(lat_src, lon_src)])
seg_d  = np.hypot(np.diff(xy_src[:, 0]), np.diff(xy_src[:, 1]))
cum_d  = np.concatenate(([0.0], np.cumsum(seg_d)))

# ── Resample to video timeline ───────────────────────────────────────────────
n_frames = int(FPS * DURATION)
t        = np.linspace(0, DURATION, n_frames)
speed    = np.interp(t, t_src, spd_src)
bx       = np.interp(t, t_src, np.cos(np.radians(brg_src)))
by       = np.interp(t, t_src, np.sin(np.radians(brg_src)))
bearing  = np.degrees(np.arctan2(by, bx)) % 360
distance = np.interp(t, t_src, cum_d)

speed_kmh   = speed * 3.6
max_kmh     = float(speed_kmh.max())
SMAX        = max(50.0, max_kmh * 1.1)
total_km    = float(distance[-1] / 1000)

# Turn rate (deg/s): gradient of unwrapped bearing, smoothed over ~1s
brg_uw = np.degrees(np.unwrap(np.radians(bearing)))
turn   = np.gradient(brg_uw) * FPS
W      = max(3, FPS // 2 | 1)
kernel = np.ones(W) / W
turn   = np.convolve(np.pad(turn, W // 2, mode="edge"), kernel, mode="valid")
turn   = turn[: n_frames] if len(turn) >= n_frames else np.pad(
    turn, (0, n_frames - len(turn)), mode="edge"
)
TURN_MAX = 25.0   # deg/s scale (~highway sweeping turn)

# ── Pre-render speed chart base ──────────────────────────────────────────────
CHART_X, CHART_Y = 18, 240
CHART_W, CHART_H = WIDTH - 36, 95

def build_chart_base(w, h, ys, ymax):
    img = np.full((h, w, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), LINE, 1)
    for gv in range(10, int(ymax) + 1, 10):
        y = int(h - (gv / ymax) * (h - 4)) - 2
        cv2.line(img, (1, y), (w - 1, y), (235, 235, 235), 1)
        cv2.putText(img, f"{gv}", (3, y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
    pts = np.array([
        (int(i / (len(ys) - 1) * (w - 1)),
         int(h - (ys[i] / ymax) * (h - 4)) - 2)
        for i in range(len(ys))
    ], dtype=np.int32)
    poly = np.vstack([pts, [[w - 1, h - 1], [0, h - 1]]])
    ov = img.copy()
    cv2.fillPoly(ov, [poly], (215, 200, 175))
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    cv2.polylines(img, [pts], False, ACC, 2, cv2.LINE_AA)
    return img, pts

chart_base, chart_pts = build_chart_base(CHART_W, CHART_H, speed_kmh, SMAX)


def draw_gauge(frame, cx, cy, r, value, vmax):
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, 360, (225, 225, 225), 14, cv2.LINE_AA)
    pct = max(0.0, min(1.0, value / vmax))
    if pct > 0:
        end = 180 + pct * 180
        cv2.ellipse(frame, (cx, cy), (r, r), 0, 180, end,
                    speed_to_bgr(pct), 14, cv2.LINE_AA)
    for k in range(0, 6):
        a = 180 + k * 36
        rad = math.radians(a)
        x1 = int(cx + (r - 10) * math.cos(rad)); y1 = int(cy + (r - 10) * math.sin(rad))
        x2 = int(cx + (r + 10) * math.cos(rad)); y2 = int(cy + (r + 10) * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), (155, 155, 155), 2, cv2.LINE_AA)
    rad = math.radians(180 + pct * 180)
    nx = int(cx + (r - 6) * math.cos(rad)); ny = int(cy + (r - 6) * math.sin(rad))
    cv2.line(frame, (cx, cy), (nx, ny), INK, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 7, INK, -1, cv2.LINE_AA)


def draw_turn_bar(frame, cx, cy, w, h, value, vmax):
    half_w = w // 2
    cv2.rectangle(frame, (cx - half_w, cy - h // 2),
                  (cx + half_w, cy + h // 2), (235, 235, 235), -1)
    cv2.rectangle(frame, (cx - half_w, cy - h // 2),
                  (cx + half_w, cy + h // 2), (175, 175, 175), 1)
    cv2.line(frame, (cx, cy - h // 2 - 4),
             (cx, cy + h // 2 + 4), (100, 100, 100), 2)
    pct = max(-1.0, min(1.0, value / vmax))
    bar = int(half_w * abs(pct))
    if pct < -0.02:
        cv2.rectangle(frame, (cx - bar, cy - h // 2 + 2),
                      (cx, cy + h // 2 - 2), LEFT_C, -1)
    elif pct > 0.02:
        cv2.rectangle(frame, (cx, cy - h // 2 + 2),
                      (cx + bar, cy + h // 2 - 2), RIGHT_C, -1)


# ── Render ───────────────────────────────────────────────────────────────────
print(f"Rendering {n_frames} frames...")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(str(OUTPUT), fourcc, float(FPS), (WIDTH, HEIGHT))

for i in range(n_frames):
    f = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)

    # Title bar
    cv2.rectangle(f, (0, 0), (WIDTH, 46), (255, 255, 255), -1)
    cv2.line(f, (0, 46), (WIDTH, 46), LINE, 1)
    cv2.putText(f, "DRIVE  STATS", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 2, cv2.LINE_AA)
    cv2.putText(f, f"{t[i]:5.1f} / {DURATION:.0f}s",
                (255, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SUB, 1, cv2.LINE_AA)

    # Speedometer
    gcx, gcy, gr = 200, 175, 82
    draw_gauge(f, gcx, gcy, gr, speed_kmh[i], SMAX)
    txt = f"{speed_kmh[i]:.0f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    cv2.putText(f, txt, (gcx - tw // 2, gcy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, INK, 3, cv2.LINE_AA)
    cv2.putText(f, "km/h", (gcx - 22, gcy + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, SUB, 1, cv2.LINE_AA)
    cv2.putText(f, "0", (gcx - gr - 14, gcy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, SUB, 1, cv2.LINE_AA)
    cv2.putText(f, f"{SMAX:.0f}", (gcx + gr - 4, gcy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, SUB, 1, cv2.LINE_AA)

    # Speed chart
    cv2.putText(f, "Speed (km/h) over trip", (CHART_X, CHART_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, SUB, 1, cv2.LINE_AA)
    f[CHART_Y: CHART_Y + CHART_H, CHART_X: CHART_X + CHART_W] = chart_base.copy()
    px = CHART_X + int(i / (n_frames - 1) * (CHART_W - 1))
    py = chart_pts[i][1] + CHART_Y
    cv2.line(f, (px, CHART_Y + 1), (px, CHART_Y + CHART_H - 1),
             (60, 76, 231), 2)
    cv2.circle(f, (px, py), 4, (60, 76, 231), -1, cv2.LINE_AA)

    # Turn-rate
    tcy = 380
    cv2.putText(f, "Turn rate (deg/s)", (CHART_X, tcy - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, SUB, 1, cv2.LINE_AA)
    draw_turn_bar(f, WIDTH // 2, tcy, WIDTH - 60, 22, turn[i], TURN_MAX)
    cv2.putText(f, "L", (16, tcy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEFT_C, 2, cv2.LINE_AA)
    cv2.putText(f, "R", (WIDTH - 25, tcy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RIGHT_C, 2, cv2.LINE_AA)
    tsign = "L" if turn[i] < -0.5 else ("R" if turn[i] > 0.5 else "-")
    cv2.putText(f, f"{abs(turn[i]):4.1f} {tsign}", (WIDTH // 2 - 30, tcy + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 2, cv2.LINE_AA)

    # Footer
    fy = HEIGHT - 18
    cv2.line(f, (12, fy - 30), (WIDTH - 12, fy - 30), LINE, 1)
    cv2.putText(f, f"Hdg: {bearing[i]:3.0f}deg  {cardinal(bearing[i])}",
                (14, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK, 1, cv2.LINE_AA)
    cv2.putText(f, f"Dist: {distance[i]/1000:.2f} / {total_km:.2f} km",
                (14, fy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK, 1, cv2.LINE_AA)
    cv2.putText(f, f"Max: {max_kmh:.0f} km/h",
                (250, fy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK, 1, cv2.LINE_AA)

    out.write(f)
    if (i + 1) % (FPS * 30) == 0:
        print(f"  {i+1}/{n_frames}")

out.release()
print(f"Done: {OUTPUT}")
