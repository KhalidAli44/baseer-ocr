import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from config import AH_HEIGHT_MIN, AH_HEIGHT_MAX, AH_AREA_MIN

KDE_BANDWIDTH      = 0.6
PEAK_PROMINENCE    = 0.15
MAX_ASSIGN_DIST    = 1.2
ORPHAN_MIN_MEMBERS = 3
LINE_PAD_FRACTION  = 0.15


def _estimate_ah(binary):
    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    heights = [stats[i][3] for i in range(1, n) if stats[i][4] > 5]
    return float(np.median(heights)) if heights else 20.0


def detect_lines(binary, ah=None):
    if ah is None:
        ah = _estimate_ah(binary)

    inv = 255 - binary
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    components = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (AH_HEIGHT_MIN * ah < h < AH_HEIGHT_MAX * ah
                and area > AH_AREA_MIN * ah * ah):
            components.append((y + h / 2.0, y, y + h))

    if not components:
        return [], ah

    h_img  = binary.shape[0]
    votes  = np.zeros(h_img, dtype=float)
    sigma  = ah * KDE_BANDWIDTH

    for cy, _, _ in components:
        votes[int(np.clip(cy, 0, h_img - 1))] += 1.0

    density  = gaussian_filter1d(votes, sigma=sigma)
    min_dist = int(ah * 0.8)
    peaks, _ = find_peaks(density,
                          distance=min_dist,
                          height=density.max() * PEAK_PROMINENCE)

    if len(peaks) == 0:
        return [], ah

    max_dist  = ah * MAX_ASSIGN_DIST
    groups    = {p: [] for p in peaks}
    orphans   = []

    for comp in components:
        cy, y1, y2 = comp
        dists      = np.abs(peaks - cy)
        nearest_i  = int(np.argmin(dists))
        if dists[nearest_i] <= max_dist:
            groups[peaks[nearest_i]].append(comp)
        else:
            orphans.append(comp)

    orphan_lines = []
    if orphans:
        orphans.sort(key=lambda c: c[0])
        cluster, cluster_y2 = [orphans[0]], orphans[0][2]
        for comp in orphans[1:]:
            if comp[1] <= cluster_y2 + ah * 0.5:
                cluster.append(comp)
                cluster_y2 = max(cluster_y2, comp[2])
            else:
                if len(cluster) >= ORPHAN_MIN_MEMBERS:
                    orphan_lines.append(cluster)
                cluster, cluster_y2 = [comp], comp[2]
        if len(cluster) >= ORPHAN_MIN_MEMBERS:
            orphan_lines.append(cluster)

    pad      = max(1, int(ah * LINE_PAD_FRACTION))
    all_lines = []

    for members in groups.values():
        if not members:
            continue
        ls = max(0,     min(c[1] for c in members) - pad)
        le = min(h_img, max(c[2] for c in members) + pad)
        all_lines.append((ls, le))

    for members in orphan_lines:
        ls = max(0,     min(c[1] for c in members) - pad)
        le = min(h_img, max(c[2] for c in members) + pad)
        all_lines.append((ls, le))

    all_lines.sort(key=lambda l: l[0])
    return all_lines, ah