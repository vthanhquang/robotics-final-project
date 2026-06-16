"""
Inspect the VinUni / Ocean Park CommonRoad map to ground realistic scenarios:
  * typical lane width (for realistic lateral offsets),
  * how many parallel lanes exist (multi-lane >2 streets),
  * real crossroads near the GPS route (perpendicular lanelet crossings),
  * where these sit along the straightest reference window used by the suite.

Run:  python src/inspect_map.py
"""

import math
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

from scenario_suite import load_gps_xy, straightest_window, REF_WINDOW_M

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "data" / "maps" / "VinUni-1_2-T1.xml"
NEAR_M = 70.0


def seg_intersect(p1, p2, p3, p4):
    d1 = p2 - p1; d2 = p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * d1
    return None


def main():
    scenario, _ = CommonRoadFileReader(str(XML_PATH)).open()
    ln = scenario.lanelet_network
    lanelets = list(ln.lanelets)
    print(f"Total lanelets in map: {len(lanelets)}")
    print(f"API intersections modeled: {len(ln.intersections)}")

    # reference window the suite drives on
    xy = load_gps_xy()
    win = straightest_window(xy, REF_WINDOW_M)
    seg = np.hypot(np.diff(win[:, 0]), np.diff(win[:, 1]))
    s_win = np.concatenate([[0], np.cumsum(seg)])

    def s_along(pt):
        i = int(np.argmin(np.hypot(win[:, 0] - pt[0], win[:, 1] - pt[1])))
        return s_win[i], float(np.hypot(win[i, 0] - pt[0], win[i, 1] - pt[1]))

    wmin = win.min(0) - NEAR_M; wmax = win.max(0) + NEAR_M

    # lanelets near the reference window
    near = []
    for ll in lanelets:
        c = ll.center_vertices
        if (c[:, 0].max() < wmin[0] or c[:, 0].min() > wmax[0]
                or c[:, 1].max() < wmin[1] or c[:, 1].min() > wmax[1]):
            continue
        # at least one vertex truly within NEAR_M of the window
        d = min(s_along(v)[1] for v in c[::max(1, len(c) // 6)])
        if d < NEAR_M:
            near.append(ll)
    print(f"Lanelets near the reference window: {len(near)}")

    # lane widths (near route)
    widths = []
    for ll in near:
        w = np.hypot(ll.left_vertices[:, 0] - ll.right_vertices[:, 0],
                     ll.left_vertices[:, 1] - ll.right_vertices[:, 1])
        widths.append(float(np.mean(w)))
    widths = np.array(widths)
    print(f"Lane width near route: median {np.median(widths):.2f} m, "
          f"range {widths.min():.2f}-{widths.max():.2f} m")

    # parallel-lane clusters via adjacency (union-find over near lanelets)
    ids = {ll.lanelet_id for ll in near}
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        if b in parent:
            parent[find(a)] = find(b)

    for ll in near:
        for adj in (ll.adj_left, ll.adj_right):
            if adj in ids:
                union(ll.lanelet_id, adj)
    clusters = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"Parallel-lane cluster sizes near route (top): {sizes[:6]}")
    print(f"  -> widest road near route = {sizes[0]} parallel lanes")

    # crossroads: near-route lanelet centerlines that cross at a large angle
    def head(c):
        return math.atan2(c[-1, 1] - c[0, 1], c[-1, 0] - c[0, 0])

    crossings = []
    for a in range(len(near)):
        ca = near[a].center_vertices
        ha = head(ca)
        for b in range(a + 1, len(near)):
            cb = near[b].center_vertices
            dh = abs((head(cb) - ha + math.pi) % (2 * math.pi) - math.pi)
            if dh < math.radians(45):
                continue
            for i in range(len(ca) - 1):
                for j in range(len(cb) - 1):
                    x = seg_intersect(ca[i], ca[i + 1], cb[j], cb[j + 1])
                    if x is not None:
                        crossings.append((x, math.degrees(dh)))
                        break
                else:
                    continue
                break
    # dedupe crossing points + report those closest to the route
    uniq = []
    for x, dh in crossings:
        if all(np.hypot(x[0] - u[0][0], x[1] - u[0][1]) > 5 for u in uniq):
            uniq.append((x, dh))
    scored = sorted(((s_along(x)[1], s_along(x)[0], x, dh) for x, dh in uniq))
    print(f"\nPerpendicular lanelet crossings near route: {len(uniq)}")
    print("  closest crossroads (dist_to_route, s_along_window, angle):")
    for dist, s, x, dh in scored[:6]:
        print(f"    s={s:6.1f} m  off-route {dist:5.1f} m  angle {dh:3.0f} deg  "
              f"@({x[0]:.0f},{x[1]:.0f})")
    print(f"\nReference window length: {s_win[-1]:.0f} m")


if __name__ == "__main__":
    main()
