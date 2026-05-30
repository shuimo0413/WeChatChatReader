"""
这个文件是用来识别自己还是对方的气泡的（基于颜色识别），暂时不用管

"""


from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


DEFAULT_HSV_RANGES: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([35, 35, 45], dtype=np.uint8), np.array([92, 255, 255], dtype=np.uint8)),
    (np.array([55, 25, 55], dtype=np.uint8), np.array([85, 200, 255], dtype=np.uint8)),
)


def classify_role_bgr(
    image_bgr: np.ndarray,
    xyxy: np.ndarray | Sequence[float],
    shrink_ratio: float = 0.12,
    hsv_ranges: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    min_green_ratio: float = 0.08,
) -> tuple[str, float]:
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    mx, my = w * shrink_ratio, h * shrink_ratio
    ix1 = int(x1 + mx)
    iy1 = int(y1 + my)
    ix2 = int(x2 - mx)
    iy2 = int(y2 - my)
    if ix2 <= ix1:
        ix1, ix2 = int(x1), int(x2)
    if iy2 <= iy1:
        iy1, iy2 = int(y1), int(y2)

    H, W = image_bgr.shape[:2]
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(W, ix2), min(H, iy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return "peer", 0.0

    roi = image_bgr[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return "peer", 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ranges = hsv_ranges if hsv_ranges is not None else DEFAULT_HSV_RANGES
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    total = int(mask.size)
    green_pixels = int(cv2.countNonZero(mask))
    ratio = green_pixels / total if total else 0.0
    role = "self" if ratio >= min_green_ratio else "peer"
    return role, ratio
