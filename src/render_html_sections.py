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
BG = (0x19, 0x14, 0x0f)            # #0f1419 in BGR


def build_pages():
    html = HTML.read_text()
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
