# core/motion_detector.py
"""
Motion detection engine using MOG2 Gaussian Mixture Model background subtraction.
Based on: Stauffer & Grimson (1999) — Adaptive background mixture models for real-time tracking.
"""

import cv2
import numpy as np
from utils.config import (
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS,
    MOG2_LEARNING_RATE, MIN_CONTOUR_AREA, MAX_FOREGROUND_RATIO,
    MOTION_BLUR_KERNEL, MORPH_KERNEL_SIZE
)


class MotionDetector:
    """
    Detects motion in video frames using MOG2 background subtraction.
    Returns bounding boxes of detected motion regions.
    """

    def __init__(self):
        # Initialise MOG2 subtractor with conservative learning rate
        # to prevent slow-moving objects from being absorbed into background
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        )
        # Structuring element for morphological operations (Serra, 1982)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE
        )
        self.is_motion     = False
        self.contours      = []
        self.fg_ratio      = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> list[tuple]:
        """
        Process a single frame. Returns list of (x, y, w, h) bounding boxes
        for each motion region that exceeds the minimum contour area.

        Args:
            frame: BGR frame from VideoCapture

        Returns:
            List of bounding box tuples (x, y, w, h)
        """
        preprocessed = self._preprocess(frame)
        fg_mask      = self._apply_subtractor(preprocessed)
        cleaned_mask = self._clean_mask(fg_mask)
        boxes        = self._extract_contours(cleaned_mask, frame.shape)

        self.is_motion = len(boxes) > 0
        return boxes

    def get_foreground_mask(self) -> np.ndarray | None:
        """Return the last computed foreground mask for debug/UI display."""
        return getattr(self, "_last_mask", None)

    def reset(self):
        """Re-initialise the background model — use when switching video sources."""
        self.__init__()

    # ── Private Methods ───────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert to greyscale and apply Gaussian blur.
        Greyscale: reduces 3-channel to 1-channel, cutting computation by ~3x.
        Gaussian blur: suppresses high-frequency pixel noise before subtraction.
        """
        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, MOTION_BLUR_KERNEL, 0)
        return blurred

    def _apply_subtractor(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Apply MOG2 subtractor. Returns binary foreground mask.
        Shadows are detected (value=127) and excluded (thresholded to 0).
        """
        fg_mask = self._subtractor.apply(preprocessed, learningRate=MOG2_LEARNING_RATE)

        # Exclude shadows (127) — keep only confirmed foreground (255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Guard: if >60% of frame is foreground, it's a lighting change — skip
        self.fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        if self.fg_ratio > MAX_FOREGROUND_RATIO:
            return np.zeros_like(fg_mask)

        return fg_mask

    def _clean_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening (erosion then dilation) to remove noise,
        then closing (dilation then erosion) to fill holes within objects.
        """
        # Opening: removes small isolated noise blobs
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._morph_kernel, iterations=2)
        # Closing: fills gaps and holes within detected objects
        closed = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

        self._last_mask = closed
        return closed

    def _extract_contours(self, mask: np.ndarray, frame_shape: tuple) -> list[tuple]:
        """
        Extract contours from cleaned mask. Filter by minimum area.
        Returns list of (x, y, w, h) bounding boxes.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue                      # Skip noise

            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        return boxes