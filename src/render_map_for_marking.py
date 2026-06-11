"""
Render the VinUni CommonRoad map as a high-res PNG with:
  * lanelet network in grey
  * 100 m coordinate grid + pixel-coordinate axis labels (every 200 px)
  * the *current* mis-calibrated GPS track in red (centroid-shifted)
  * the *raw* GPS track shifted only by the map-centroid offset, for reference
  * GPS path markers labelled with seconds-elapsed (every 20 s) so you can pin
    a real-world landmark to a specific point of the drive

Open the PNG, mark the *correct* pixel positions for:
    A) GPS start                  (t = 0 s)
    B) GPS first left turn        (~t = 105 s in the raw data)
Tell me the (px, py) pairs and I'll fit a similarity transform
(translation + rotation + uniform scale) and re-render.
"""

import csv, math
from pathlib import Path

import cv2
import numpy as np

from commonroad.common.file_reader import CommonRoadFileReader

ROOT     = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "maps" / "VinUni-1_1-T1.xml"
GPS_PATH = ROOT / "data" / "raw" / "sensors" / "Location.csv"
OUT_PNG  = ROOT / "outputs" / "map_for_marking.png"

PX_PER_M = 2          # 2 px / metre  ->  ~4670 x 2970 image
PAD_M    = 80
R_EARTH  = 6_378_137.0


def wgs84_to_m(lat, lon):
    x = lon * math.pi * R_EARTH / 180
    y = math.log(math.tan(math.pi / 4 + lat * math.pi / 360)) * R_EARTH
    return x, y


# ── 1. Load GPS ───────────────────────────────────────────────────────────────
rows = []
with open(GPS_PATH, newline="") as f:
    for r in csv.DictReader(f):
        rows.append((
            float(r["seconds_elapsed"]),
            float(r["latitude"]),
            float(r["longitude"]),
        ))
rows.sort(key=lambda r: r[0])
t_src = np.array([r[0] for r in rows]); t_src -= t_src.min()
xy_raw = np.array([wgs84_to_m(la, lo) for _, la, lo in rows])

# ── 2. Load CommonRoad map ────────────────────────────────────────────────────
scenario, _ = CommonRoadFileReader(str(XML_PATH)).open()

# Map extent from the lanelets themselves (no hard-coded numbers)
all_pts = np.vstack([
    np.vstack([ll.left_vertices, ll.right_vertices])
    for ll in scenario.lanelet_network.lanelets
])
X_MIN, Y_MIN = all_pts.min(axis=0) - PAD_M
X_MAX, Y_MAX = all_pts.max(axis=0) + PAD_M

W = int((X_MAX - X_MIN) * PX_PER_M)
H = int((Y_MAX - Y_MIN) * PX_PER_M)
print(f"Map: {W} x {H} px  ({X_MAX - X_MIN:.0f} x {Y_MAX - Y_MIN:.0f} m)")

def m_to_px(x, y):
    return int((x - X_MIN) * PX_PER_M), int((Y_MAX - y) * PX_PER_M)


# ── 3. Apply current centroid-shift to GPS (matches cr_gps_drive.py) ──────────
map_cx = (X_MIN + X_MAX) / 2
map_cy = (Y_MIN + Y_MAX) / 2
gps_cx = (xy_raw[:, 0].min() + xy_raw[:, 0].max()) / 2
gps_cy = (xy_raw[:, 1].min() + xy_raw[:, 1].max()) / 2
dx, dy = map_cx - gps_cx, map_cy - gps_cy
xy_shift = xy_raw + np.array([dx, dy])
print(f"Current centroid-shift: dx={dx:.1f} m  dy={dy:.1f} m")

# ── 4. Draw ───────────────────────────────────────────────────────────────────
img = np.full((H, W, 3), (252, 250, 248), dtype=np.uint8)

# 100 m fine grid
for gx in np.arange(math.ceil(X_MIN / 100) * 100, X_MAX, 100):
    px, _ = m_to_px(gx, Y_MIN)
    cv2.line(img, (px, 0), (px, H), (228, 226, 224), 1)
for gy in np.arange(math.ceil(Y_MIN / 100) * 100, Y_MAX, 100):
    _, py = m_to_px(X_MIN, gy)
    cv2.line(img, (0, py), (W, py), (228, 226, 224), 1)

# 200 px coarse grid + labels — this is what you read off the image viewer
for px in range(0, W, 200):
    cv2.line(img, (px, 0), (px, H), (180, 180, 200), 1)
    cv2.putText(img, f"x={px}", (px + 4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 130), 1, cv2.LINE_AA)
for py in range(0, H, 200):
    cv2.line(img, (0, py), (W, py), (180, 180, 200), 1)
    cv2.putText(img, f"y={py}", (6, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 130), 1, cv2.LINE_AA)

# Lanelets — fill + bounds
for ll in scenario.lanelet_network.lanelets:
    poly = np.vstack([ll.left_vertices, ll.right_vertices[::-1]])
    poly_px = np.array([m_to_px(p[0], p[1]) for p in poly], dtype=np.int32)
    cv2.fillPoly(img, [poly_px], (215, 215, 215))
for ll in scenario.lanelet_network.lanelets:
    lv = np.array([m_to_px(p[0], p[1]) for p in ll.left_vertices],  dtype=np.int32)
    rv = np.array([m_to_px(p[0], p[1]) for p in ll.right_vertices], dtype=np.int32)
    cv2.polylines(img, [lv], False, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.polylines(img, [rv], False, (80, 80, 80), 2, cv2.LINE_AA)

# Current (centroid-shifted) GPS — red
gps_px = np.array([m_to_px(p[0], p[1]) for p in xy_shift], dtype=np.int32)
cv2.polylines(img, [gps_px], False, (40, 40, 220), 3, cv2.LINE_AA)

# Time-tick markers every 20 s with labels
for i, (sec, lat, lon) in enumerate(rows):
    if i == 0 or i == len(rows) - 1 or int(sec) % 20 == 0 and int(sec) != int(rows[i-1][0]):
        x, y = xy_shift[i]
        px, py = m_to_px(x, y)
        cv2.circle(img, (px, py), 7, (255, 255, 255), -1)
        cv2.circle(img, (px, py), 7, (20, 20, 160), 2, cv2.LINE_AA)
        label = "START" if i == 0 else ("END" if i == len(rows) - 1 else f"{int(sec)}s")
        cv2.putText(img, label, (px + 10, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 160), 2, cv2.LINE_AA)

# Header
cv2.rectangle(img, (12, 12), (640, 90), (255, 255, 255), -1)
cv2.rectangle(img, (12, 12), (640, 90), (140, 140, 160), 1)
cv2.putText(img, "VinUni CommonRoad map  +  current GPS (red)",
            (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
cv2.putText(img, "Mark CORRECT pixel positions of START and FIRST LEFT TURN.",
            (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 100), 1, cv2.LINE_AA)

cv2.imwrite(str(OUT_PNG), img)
print(f"Wrote: {OUT_PNG}")
print(f"GPS time range: {t_src.min():.0f}..{t_src.max():.0f} s ({len(rows)} samples)")
