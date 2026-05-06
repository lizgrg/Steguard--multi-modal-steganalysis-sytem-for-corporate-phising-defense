# core/video_handler.py
import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque
from utils.config import (
    OUTPUT_DIR, CLIP_BUFFER_BEFORE, CLIP_BUFFER_AFTER,
    VIDEO_CODEC, CLIP_FPS, FRAME_WIDTH, FRAME_HEIGHT
)


class VideoSource:
    def __init__(self, source=0):
        self.source = source
        self._cap   = None
        self.fps    = CLIP_FPS
        self.width  = FRAME_WIDTH
        self.height = FRAME_HEIGHT

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or CLIP_FPS
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap:
            self._cap.release()

    def is_file(self):
        return isinstance(self.source, str)


class IncidentRecorder:
    def __init__(self, fps, width, height):
        self.fps         = max(fps, 1)
        self.width       = width
        self.height      = height
        self._pre_buffer = deque(maxlen=CLIP_BUFFER_BEFORE)
        self._writer     = None
        self._post_count = 0
        self._recording  = False
        self._clip_path  = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def feed(self, frame: np.ndarray, motion: bool, label: str = "UNKNOWN") -> bool:
        self._pre_buffer.append(frame.copy())

        if motion and not self._recording:
            self._start_recording(label)

        if self._recording:
            try:
                self._writer.write(frame)
            except Exception:
                pass
            if not motion:
                self._post_count += 1
                if self._post_count >= CLIP_BUFFER_AFTER:
                    self._stop_recording()
            else:
                self._post_count = 0

        return self._recording

    def get_saved_clips(self) -> list:
        if not os.path.exists(OUTPUT_DIR):
            return []
        return sorted(
            [f for f in os.listdir(OUTPUT_DIR)
             if f.endswith(".avi") or f.endswith(".mp4")],
            reverse=True
        )

    def stop(self):
        if self._recording:
            self._stop_recording()

    def _start_recording(self, label: str):
        ts              = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._clip_path = os.path.join(OUTPUT_DIR, f"incident_{label}_{ts}.avi")
        fourcc          = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer    = cv2.VideoWriter(
            self._clip_path, fourcc, self.fps, (self.width, self.height)
        )
        for f in self._pre_buffer:
            try:
                self._writer.write(f)
            except Exception:
                pass
        self._recording  = True
        self._post_count = 0
        print(f"[REC] Started: {self._clip_path}")

    def _stop_recording(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        self._recording = False
        print(f"[REC] Saved: {self._clip_path}")