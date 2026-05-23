import cv2
import numpy as np

from config import AH_HEIGHT_MIN, AH_HEIGHT_MAX, AH_AREA_MIN

WORD_GAP_SCALE = 0.75


def _extract_blobs(strip_inv, ah):
    n, _, stats, _ = cv2.connectedComponentsWithStats(strip_inv, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (AH_HEIGHT_MIN * ah < h < AH_HEIGHT_MAX * ah
                and area > AH_AREA_MIN * ah * ah):
            blobs.append((x, y, x + w, y + h))
    return sorted(blobs, key=lambda b: b[0])


def _estimate_word_gap_threshold(blobs, ah):
    if len(blobs) < 2:
        return ah * WORD_GAP_SCALE

    gaps = []
    for i in range(1, len(blobs)):
        gap = blobs[i][0] - blobs[i - 1][2]
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return ah * WORD_GAP_SCALE

    gaps = np.array(sorted(gaps), dtype=np.float32)

    if len(gaps) < 4:
        return float(np.max(gaps)) * 0.75

    median_gap = float(np.median(gaps))
    large_gaps = gaps[gaps > median_gap]
    small_gaps = gaps[gaps <= median_gap]

    if len(large_gaps) == 0:
        return median_gap * 2.0

    mean_small = float(np.mean(small_gaps)) if len(small_gaps) > 0 else 0.0
    mean_large = float(np.mean(large_gaps))

    threshold = (mean_small + mean_large) / 2.0
    return max(threshold, ah * 0.3)


def _group_blobs_into_words(blobs, threshold):
    if not blobs:
        return []

    words, current = [], [blobs[0]]
    for blob in blobs[1:]:
        gap = blob[0] - current[-1][2]
        if gap > threshold:
            words.append(current)
            current = [blob]
        else:
            current.append(blob)
    words.append(current)

    return [
        (min(b[0] for b in w), min(b[1] for b in w),
         max(b[2] for b in w), max(b[3] for b in w))
        for w in words
    ]


def detect_words(binary, ls, le, ah):
    strip_inv = 255 - binary[ls:le, :]
    blobs     = _extract_blobs(strip_inv, ah)

    if not blobs:
        return []

    threshold = _estimate_word_gap_threshold(blobs, ah)
    words     = _group_blobs_into_words(blobs, threshold)

    return [(x1, ls + y1, x2, ls + y2) for x1, y1, x2, y2 in words]