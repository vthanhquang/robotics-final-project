"""
Render apex_architecture.html into per-section PNGs (Chrome headless), preserving
the original dark design, for use as slide images.

Out: outputs/apex_arch_inherit.png, apex_arch_core.png, apex_arch_results.png,
     apex_arch_learn.png

Run:  python src/render_html_sections.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "apex_architecture.html"
OUT = ROOT / "outputs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LIGHT = True                       # render in the white/red/blue VinUni theme
BG = (255, 255, 255) if LIGHT else (0x19, 0x14, 0x0f)   # trim background (BGR)

# dark -> light (white / VinUni red+blue) replacements
_LIGHT_REPL = {
    # :root palette
    "--bg:#0f1419": "--bg:#ffffff", "--panel:#171e26": "--panel:#f5f7fa",
    "--panel2:#1d2731": "--panel2:#eef2f7", "--ink:#e8edf2": "--ink:#1d2733",
    "--mut:#9fb0c0": "--mut:#5b6b7b", "--line:#33414f": "--line:#d3dbe4",
    "--apex:#3ddc84": "--apex:#1e9e4a", "--idm:#5aa9ff": "--idm:#1f5fb2",
    "--orca:#ffb454": "--orca:#c47d10", "--frenet:#c77dff": "--frenet:#7a3fb0",
    "--learn:#ff7aa2": "--learn:#c0407a", "--bad:#ff5d5d": "--bad:#cc3b3b",
    "--good:#3ddc84": "--good:#1e9e4a",
    # hardcoded dark surfaces
    "linear-gradient(180deg,#16241c,#13201a)": "linear-gradient(180deg,#eefaf2,#e6f6ec)",
    "color:#cfe9d8;background:#0e1813": "color:#157a39;background:#eef2f7",
    # inline category-chip backgrounds
    "background:#13314f": "background:#e7effb", "background:#4a3415": "background:#fbf0dc",
    "background:#3a2452": "background:#f0e7fb", "background:#4a2233": "background:#fbe7f0",
    "background:#16341f": "background:#e7fbee", "background:#26303a": "background:#eef2f7",
}


def build_pages():
    html = HTML.read_text()
    if LIGHT:
        for a, b in _LIGHT_REPL.items():
            html = html.replace(a, b)
    style = re.search(r"<style>.*?</style>", html, re.S).group(0)
    wrap = re.search(r'<div class="wrap">(.*)</div>\s*</body>', html, re.S).group(1)
    parts = re.split(r'<!-- =+ ([A-Z /\-&]+?) =+ -->', wrap)
    header = parts[0]                         # h1 + sub
    sec = {parts[i].strip(): parts[i + 1] for i in range(1, len(parts), 2)}
    h1 = "<h1><span>APEX</span> architecture</h1>"

    def page(inner):
        return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>{style}</head>"
                f"<body><div class='wrap'>{inner}</div></body></html>")

    return {
        "apex_arch_inherit": page(header + sec["INHERITANCE"]),
        "apex_arch_core":    page(h1 + sec["APEX CORE"]),
        "apex_arch_results": page(h1 + sec["RESULTS"]),
        "apex_arch_learn":   page(h1 + sec["LEARNING-AUGMENTED MPC"]),
    }


def trim(path):
    img = cv2.imread(str(path))
    diff = np.abs(img.astype(int) - np.array(BG)).sum(2) > 40   # non-bg pixels
    rows = np.where(diff.any(1))[0]
    cols = np.where(diff.any(0))[0]
    if len(rows) and len(cols):
        pad = 24
        r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad)
        c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad)
        cv2.imwrite(str(path), img[r0:r1, c0:c1])


def main():
    pages = build_pages()
    for name, content in pages.items():
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
            f.write(content); tmp = f.name
        out = OUT / f"{name}.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1260,1700",
                        f"--screenshot={out}", f"file://{tmp}"],
                       capture_output=True)
        trim(out)
        h, w = cv2.imread(str(out)).shape[:2]
        print(f"  {out.name}  {w}x{h}")


if __name__ == "__main__":
    main()
