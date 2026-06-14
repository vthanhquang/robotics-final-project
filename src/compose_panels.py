"""
Composite the three replay views into the single 3-panel video shown in
Progress Report 2 (Fig. 1 / Fig. 3):

    [ GPS on CommonRoad map | forward camera + YOLO detections | drive stats ]

The three source views have different sizes and frame rates; this tool
resamples them onto one common timeline (no re-encoding of content, just
nearest-frame lookup) and stacks them with a title bar, per-panel captions and
an optional low/high traffic-mode badge from src/segment_traffic.py.

Middle (camera) panel:
  * if data/outputs/Car_OCP_labeled.mp4 exists, it is used directly;
  * otherwise the 5 committed labeled stills (sample_*.jpg) are shown as a
    time-synced slideshow, so the final layout is verifiable before the full
    labeled video has been rendered.

Run:  python src/compose_panels.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAP_VID   = ROOT / "outputs" / "cr_gps_drive.mp4"
STATS_VID = ROOT / "outputs" / "stats_panel.mp4"
CAM_VID   = ROOT / "data" / "outputs" / "Car_OCP_labeled.mp4"
CAM_STILLS = [(p / 100.0, ROOT / "data" / "outputs" / f"sample_{p:02d}pct.jpg")
              for p in (10, 30, 50, 70, 90)]
SEGMENTS  = ROOT / "outputs" / "traffic_segments.json"   # optional (mode badge)
OUTPUT    = ROOT / "outputs" / "replay_3panel.mp4"

DURATION   = 219.9
FPS        = 30
PANEL_H    = 480
TITLE_H    = 46
CAPTION_H  = 30
SEP        = 6
TITLE = "Vietnam-MixedTrafficSim  -  real GPS + YOLO detection + drive stats"
CAPTIONS = ["GPS on CommonRoad map", "YOLOv8 detection + distance", "Drive stats"]
BG = (250, 248, 246)


class Source:
    """A panel source resampled to the common timeline by timestamp."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or FPS
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._idx = -1
        self._frame = None

    def at(self, t):
        idx = min(self.n - 1, max(0, int(round(t * self.fps))))
        if idx != self._idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = self.cap.read()
            if ok:
                self._frame = fr
            self._idx = idx
        return self._frame


class StillSlideshow:
    """Fallback camera panel: hold each labeled still until the next timestamp."""

    def __init__(self, stills, duration):
        self.items = sorted((frac * duration, cv2.imread(str(p)))
                            for frac, p in stills if p.exists())
        self.duration = duration

    def at(self, t):
        cur = self.items[0][1]
        for ts, img in self.items:
            if t >= ts:
                cur = img
            else:
                break
        return cur


def fit(img, h, w):
    """Resize to height h preserving aspect, then centre on a w-wide canvas."""
    if img is None:
        canvas = np.full((h, w, 3), (235, 235, 235), np.uint8)
        cv2.putText(canvas, "camera panel: run detection", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2, cv2.LINE_AA)
        return canvas
    ih, iw = img.shape[:2]
    nw = int(iw * h / ih)
    r = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), BG, np.uint8)
    x0 = max(0, (w - nw) // 2)
    r = r[:, :w] if nw > w else r
    canvas[:, x0:x0 + r.shape[1]] = r
    return canvas


def load_segments():
    if not SEGMENTS.exists():
        return None
    return json.loads(SEGMENTS.read_text())


def mode_at(segments, t):
    if not segments:
        return None
    for seg in segments:
        if seg["t_start"] <= t < seg["t_end"]:
            return seg["mode"]
    return None


def main():
    map_src = Source(MAP_VID)
    stats_src = Source(STATS_VID)
    if CAM_VID.exists():
        cam_src, cam_kind = Source(CAM_VID), "labeled video"
    else:
        cam_src, cam_kind = StillSlideshow(CAM_STILLS, DURATION), "labeled stills (slideshow)"
    segments = load_segments()
    print(f"Camera panel source: {cam_kind}")
    print(f"Traffic-mode badge: {'on' if segments else 'off (no traffic_segments.json)'}")

    # panel widths sized from each source aspect at PANEL_H
    widths = [853, 760, 427]                       # map, camera, stats columns
    total_w = sum(widths) + 2 * SEP
    out_h = TITLE_H + PANEL_H + CAPTION_H
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(FPS), (total_w, out_h))

    n_frames = int(DURATION * FPS)
    for i in range(n_frames):
        t = i / FPS
        panels = [fit(map_src.at(t), PANEL_H, widths[0]),
                  fit(cam_src.at(t), PANEL_H, widths[1]),
                  fit(stats_src.at(t), PANEL_H, widths[2])]
        sep = np.full((PANEL_H, SEP, 3), (210, 210, 210), np.uint8)
        body = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

        # title bar
        title = np.full((TITLE_H, total_w, 3), (40, 40, 40), np.uint8)
        cv2.putText(title, TITLE, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (245, 245, 245), 1, cv2.LINE_AA)
        clock = f"t = {t:5.1f} / {DURATION:.0f} s"
        cv2.putText(title, clock, (total_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
        # mode badge
        m = mode_at(segments, t)
        if m:
            col = (60, 170, 60) if m == "low" else (50, 50, 210)
            cv2.rectangle(title, (total_w - 360, 10), (total_w - 210, 36), col, -1)
            cv2.putText(title, f"{m.upper()} TRAFFIC", (total_w - 352, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # captions
        cap = np.full((CAPTION_H, total_w, 3), BG, np.uint8)
        xs = [widths[0] // 2, widths[0] + SEP + widths[1] // 2,
              widths[0] + widths[1] + 2 * SEP + widths[2] // 2]
        for x, c in zip(xs, CAPTIONS):
            (tw, _), _ = cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(cap, c, (x - tw // 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 90, 90), 1, cv2.LINE_AA)

        writer.write(np.vstack([title, body, cap]))
        if (i + 1) % (FPS * 30) == 0:
            print(f"  {i+1}/{n_frames}  ({t:.0f}s)")

    writer.release()
    print(f"\nDone: {OUTPUT}  ({total_w}x{out_h} @ {FPS}fps)")


if __name__ == "__main__":
    main()
