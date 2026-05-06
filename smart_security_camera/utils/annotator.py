# utils/annotator.py
"""
Frame annotation — draws bounding boxes, labels, centroids,
timestamps, and system status overlay onto frames.
"""

import cv2
import numpy as np
from datetime import datetime
from utils.config import ALERT_COLOURS


def hex_to_bgr(hex_colour: str) -> tuple:
    """Convert hex colour string to OpenCV BGR tuple."""
    h  = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


class Annotator:
    """Draws all visual overlays onto video frames."""

    def draw_tracked_objects(self, frame: np.ndarray, tracked_objects: list[dict]) -> np.ndarray:
        """Draw bounding boxes, labels, and centroids for all tracked objects."""
        for obj in tracked_objects:
            label    = obj.get("label", "UNKNOWN")
            obj_id   = obj.get("id", 0)
            bbox     = obj.get("bbox")
            centroid = obj.get("centroid")
            colour   = hex_to_bgr(ALERT_COLOURS.get(label, ALERT_COLOURS["UNKNOWN"]))

            if bbox:
                x, y, w, h = bbox
                # Dashed rectangle effect (draw corners)
                self._draw_corner_box(frame, x, y, w, h, colour)
                # Label background
                label_text = f"{label} #{obj_id}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colour, -1)
                cv2.putText(frame, label_text, (x + 3, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            if centroid is not None:
                cx, cy = centroid
                cv2.circle(frame, (int(cx), int(cy)), 4, colour, -1)

        return frame

    def draw_status_bar(self, frame: np.ndarray, fps: float,
                        motion: bool, object_count: int) -> np.ndarray:
        """Draw bottom status bar with timestamp, FPS, and object count."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        ts      = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        status  = f"{ts}  |  FPS: {fps:.0f}  |  Objects: {object_count}"
        colour  = (0, 200, 150) if motion else (180, 180, 180)
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)
        return frame

    def draw_alert_flash(self, frame: np.ndarray) -> np.ndarray:
        """Flash a red border on the frame when motion is first detected."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 6)
        cv2.putText(frame, "! MOTION DETECTED", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 3)
        return frame

    def draw_fg_mask_overlay(self, frame: np.ndarray,
                             mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Overlay foreground mask as a semi-transparent green tint."""
        if mask is None:
            return frame
        coloured        = np.zeros_like(frame)
        coloured[:, :, 1] = mask   # Green channel only
        cv2.addWeighted(coloured, alpha, frame, 1.0, 0, frame)
        return frame

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _draw_corner_box(self, frame, x, y, w, h, colour, length=20, thickness=2):
        """Draw a corner-only bounding box (more elegant than solid rectangle)."""
        x2, y2 = x + w, y + h
        pts = [
            # Top-left
            ((x, y + length), (x, y), (x + length, y)),
            # Top-right
            ((x2 - length, y), (x2, y), (x2, y + length)),
            # Bottom-left
            ((x, y2 - length), (x, y2), (x + length, y2)),
            # Bottom-right
            ((x2 - length, y2), (x2, y2), (x2, y2 - length)),
        ]
        for p1, corner, p2 in pts:
            cv2.line(frame, p1, corner, colour, thickness)
            cv2.line(frame, corner, p2, colour, thickness)