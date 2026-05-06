# core/object_tracker.py
"""
Centroid-based object tracker.
Assigns persistent IDs to detected objects across frames.
Inspired by: Bewley et al. (2016) SORT — Simple Online and Realtime Tracking.
"""

import numpy as np
from collections import OrderedDict
from utils.config import MAX_DISAPPEARED, MAX_DISTANCE


class CentroidTracker:
    """
    Tracks objects by computing centroids of bounding boxes and
    matching them across frames using Euclidean distance.
    """

    def __init__(self):
        self.next_id      = 0
        self.objects      = OrderedDict()   # id → centroid (cx, cy)
        self.disappeared  = OrderedDict()   # id → frames since last seen
        self.bboxes       = OrderedDict()   # id → (x, y, w, h)
        self.labels       = OrderedDict()   # id → "HUMAN" | "VEHICLE" | "UNKNOWN"

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, boxes: list[tuple], labels: list[str] | None = None) -> OrderedDict:
        """
        Update tracker with new bounding boxes from current frame.

        Args:
            boxes:  List of (x, y, w, h) bounding boxes
            labels: Corresponding classification labels (optional)

        Returns:
            OrderedDict of {object_id: (cx, cy)}
        """
        if labels is None:
            labels = ["UNKNOWN"] * len(boxes)

        # No detections — mark all objects as disappeared
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > MAX_DISAPPEARED:
                    self._deregister(obj_id)
            return self.objects

        # Compute centroids for input boxes
        input_centroids = np.array([
            (x + w // 2, y + h // 2) for (x, y, w, h) in boxes
        ])

        # No existing objects — register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self._register(centroid, boxes[i], labels[i])
            return self.objects

        # Match existing objects to input centroids
        self._match(input_centroids, boxes, labels)
        return self.objects

    def get_tracked_objects(self) -> list[dict]:
        """Return list of tracked object dicts for annotation."""
        result = []
        for obj_id, centroid in self.objects.items():
            result.append({
                "id":       obj_id,
                "centroid": centroid,
                "bbox":     self.bboxes.get(obj_id),
                "label":    self.labels.get(obj_id, "UNKNOWN"),
            })
        return result

    # ── Private Methods ───────────────────────────────────────────────────────

    def _register(self, centroid, bbox, label):
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.bboxes[self.next_id]      = bbox
        self.labels[self.next_id]      = label
        self.next_id += 1

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        self.bboxes.pop(obj_id, None)
        self.labels.pop(obj_id, None)

    def _match(self, input_centroids, boxes, labels):
        obj_ids       = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        # Euclidean distance matrix: existing objects × new detections
        D = np.linalg.norm(obj_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

        # Greedy matching: smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > MAX_DISTANCE:
                continue

            obj_id = obj_ids[row]
            self.objects[obj_id]     = input_centroids[col]
            self.bboxes[obj_id]      = boxes[col]
            self.labels[obj_id]      = labels[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in set(range(len(obj_ids))) - used_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > MAX_DISAPPEARED:
                self._deregister(obj_id)

        # Register new unmatched detections
        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col], boxes[col], labels[col])