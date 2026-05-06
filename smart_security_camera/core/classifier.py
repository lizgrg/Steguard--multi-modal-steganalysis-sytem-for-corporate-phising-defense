# core/classifier.py
"""
Object classifier — CSY3058 Smart Security Camera
Primary: contour size + aspect ratio heuristics (fast, webcam-optimised)
Secondary: HOG+SVM for full-body confirmation when box is large enough
Reference: Dalal & Triggs (2005) HOG descriptor for human detection.
"""

import cv2
from utils.config import (
    HOG_WIN_STRIDE, HOG_PADDING, HOG_SCALE, HOG_HIT_THRESHOLD,
    VEHICLE_MIN_AREA, HUMAN_MIN_ASPECT
)

# Webcam-optimised thresholds
HUMAN_MIN_AREA   = 3000     # px² — minimum blob to consider as human
HUMAN_MAX_AREA   = 85000    # px² — above this, likely vehicle


class ObjectClassifier:
    """
    Classifies detected motion regions as HUMAN, VEHICLE, or UNKNOWN.
    Uses a tiered approach: size heuristic first, HOG confirmation second.
    """

    def __init__(self):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def classify(self, frame, boxes: list) -> list:
        return [self._classify_box(frame, box) for box in boxes]

    def _classify_box(self, frame, box: tuple) -> str:
        x, y, w, h = box
        area        = w * h
        aspect      = h / w if w > 0 else 0

        # ── Tier 1: Definite vehicle — very large blob ────────────────────
        if area >= VEHICLE_MIN_AREA:
            return "VEHICLE"

        # ── Tier 2: Size + shape heuristic — primary human detection ─────
        # A person seen by a webcam is typically 3000–85000 px²
        # and taller than wide (aspect > 0.6) OR filling much of the frame
        if HUMAN_MIN_AREA <= area <= HUMAN_MAX_AREA:
            if aspect >= 0.6:
                return "HUMAN"
            # Even if slightly wide, if big enough it's likely a person
            if area >= 8000:
                return "HUMAN"

        # ── Tier 3: HOG confirmation for borderline cases ─────────────────
        pad = 30
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return "UNKNOWN"

        crop = cv2.resize(crop, (64, 128))

        rects, _ = self._hog.detectMultiScale(
            crop,
            winStride=HOG_WIN_STRIDE,
            padding=HOG_PADDING,
            scale=HOG_SCALE,
            hitThreshold=HOG_HIT_THRESHOLD
        )

        if len(rects) > 0:
            return "HUMAN"

        return "UNKNOWN"