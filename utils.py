"""
utils.py
========
Helper functions: frame extraction (video / webcam / image), image preprocessing,
drawing of bounding boxes and overlays, and a small FPS meter.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

import config as cfg


# --------------------------------------------------------------------------- #
# Frame extraction
# --------------------------------------------------------------------------- #
class VideoSource:
    """Thin wrapper around cv2.VideoCapture for a video file or a webcam index."""

    def __init__(self, source):
        self.source = source
        self.is_webcam = isinstance(source, int)
        if self.is_webcam and os.name == "nt":
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)   # faster start-up on Windows
        else:
            self.cap = cv2.VideoCapture(source)
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.isOpened() else 0.0
        self.fps = float(fps) if fps and fps > 1 else 30.0
        n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) if self.cap.isOpened() else 0
        self.total_frames = int(n) if (n and n > 0 and not self.is_webcam) else 0

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def seek(self, frame_index: int) -> None:
        if not self.is_webcam and frame_index > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def rewind(self) -> None:
        if not self.is_webcam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self.cap.release()


def iter_frames(source, every_n: int = 1, max_frames: Optional[int] = None):
    """Generator yielding (frame_index, frame_bgr, dt_seconds) for a video file / webcam index.
    `dt` is the time between yielded frames according to the source FPS (needed for optical flow)."""
    src = VideoSource(source)
    if not src.is_opened():
        raise IOError(f"Cannot open video source: {source!r}")
    idx, yielded = -1, 0
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            idx += 1
            if idx % max(every_n, 1) != 0:
                continue
            yield idx, frame, max(every_n, 1) / src.fps
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        src.release()


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess_frame(frame_bgr: np.ndarray, max_width: int = cfg.PROCESS_MAX_WIDTH,
                     enhance: bool = False) -> np.ndarray:
    """Resize (keeping aspect ratio) and optionally enhance contrast (CLAHE on the L channel).
    Normalisation to 0..1 and letter-boxing for the network are done inside the YOLO model."""
    h, w = frame_bgr.shape[:2]
    out = frame_bgr
    if w > max_width:
        scale = max_width / float(w)
        out = cv2.resize(out, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    if enhance:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_hazard(frame: np.ndarray, bbox, text: str, level: str) -> None:
    color = cfg.LEVEL_COLORS_BGR.get(level, (200, 200, 200))
    x1, y1, x2, y2 = [int(v) for v in bbox]
    thickness = 3 if level == "HIGH" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 8
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty + base - 2), color, -1)
    cv2.putText(frame, text, (x1 + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, level: str, action: str, score: float,
                speed_label: str) -> None:
    """Top banner with frame-level risk level, action and speed source."""
    color = cfg.LEVEL_COLORS_BGR.get(level, (90, 90, 90))
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"RISK: {level}  ->  {action}   (R = {score:.2f})", (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, speed_label, (max(w - tw - 10, 10), 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_corridor(frame: np.ndarray, width_ratio: float, horizon_ratio: float) -> None:
    h, w = frame.shape[:2]
    x1 = int(w * (0.5 - width_ratio / 2))
    x2 = int(w * (0.5 + width_ratio / 2))
    y = int(h * horizon_ratio)
    cv2.line(frame, (x1, h), (x1, y), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (x2, h), (x2, y), (200, 200, 200), 1, cv2.LINE_AA)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# FPS meter
# --------------------------------------------------------------------------- #
class FpsMeter:
    """Moving-average FPS / latency meter (real measured values, never hard-coded)."""

    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window)
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            self.times.append(now - self._last)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self.times:
            return 0.0
        avg = sum(self.times) / len(self.times)
        return 1.0 / avg if avg > 0 else 0.0