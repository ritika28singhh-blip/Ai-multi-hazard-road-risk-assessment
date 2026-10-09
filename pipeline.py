"""
pipeline.py
===========
Ties all stages together:

  Frame -> Preprocessing -> YOLO detection -> Feature extraction -> Context estimation
        -> Multi-Hazard Risk Fusion -> Dynamic risk score -> LOW/MEDIUM/HIGH -> MONITOR/CAUTION/WARNING
        -> annotated frame + structured results for the dashboard.

The detector is injected, so the pipeline can be unit-tested with a stub detector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config as cfg
from detector import Detection
from distance_estimator import CameraModel, DistanceEstimate, DistanceEstimator
from risk_engine import FrameRisk, RiskConfig, RiskEngine, RiskResult
from speed_estimator import SpeedReading
from utils import draw_banner, draw_corridor, draw_hazard, preprocess_frame


@dataclass
class HazardRecord:
    """Everything known about one detected hazard (features + context + risk)."""
    category: str
    raw_label: str
    confidence: float
    bbox: tuple
    distance: DistanceEstimate
    severity: float
    in_path: bool
    risk: RiskResult


@dataclass
class FrameResult:
    annotated_bgr: np.ndarray
    hazards: List[HazardRecord]
    frame_risk: FrameRisk
    speed: SpeedReading
    timings_ms: dict = field(default_factory=dict)

    @property
    def top_hazard(self) -> Optional[HazardRecord]:
        i = self.frame_risk.top_index
        return None if i is None else self.hazards[i]


class RiskPipeline:
    def __init__(self, detector, risk_config: Optional[RiskConfig] = None,
                 camera: Optional[CameraModel] = None):
        self.detector = detector
        self.risk_engine = RiskEngine(risk_config)
        self.distance_estimator = DistanceEstimator(camera)

    # allow live re-configuration from the dashboard
    def set_risk_config(self, risk_config: RiskConfig) -> None:
        self.risk_engine = RiskEngine(risk_config)

    def set_camera(self, camera: CameraModel) -> None:
        self.distance_estimator = DistanceEstimator(camera)

    # ------------------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, speed: SpeedReading,
                conf: float = cfg.DEFAULT_CONF_THRESHOLD, iou: float = cfg.DEFAULT_IOU_THRESHOLD,
                imgsz: int = cfg.DEFAULT_IMGSZ, enhance: bool = False,
                distance_override_m: Optional[float] = None,
                draw_path_corridor: bool = False) -> FrameResult:
        t_start = time.perf_counter()

        # 1) preprocessing
        t0 = time.perf_counter()
        frame = preprocess_frame(frame_bgr, cfg.PROCESS_MAX_WIDTH, enhance)
        t_pre = (time.perf_counter() - t0) * 1000.0
        h, w = frame.shape[:2]

        # 2) detection (+ 3) feature extraction: bbox, class, confidence)
        detections, t_det = self.detector.detect(frame, conf=conf, iou=iou, imgsz=imgsz)

        # 4) context estimation + 5) risk fusion
        t0 = time.perf_counter()
        corridor = self.risk_engine.config
        records: List[HazardRecord] = []
        for d in detections:
            if distance_override_m is not None:
                dist = self.distance_estimator.simulated(distance_override_m)
            else:
                dist = self.distance_estimator.estimate(d.bbox, d.category, d.raw_label, w, h)
            cx = 0.5 * (d.bbox[0] + d.bbox[2]) / w
            in_path = abs(cx - 0.5) <= cfg.CORRIDOR_WIDTH_RATIO / 2.0
            risk = self.risk_engine.assess(
                d.category, d.confidence, dist.distance_m, speed.kmh, raw_label=d.raw_label,
                in_path=in_path, speed_is_simulated=speed.is_simulated,
                distance_is_simulated=dist.is_simulated)
            records.append(HazardRecord(
                category=d.category, raw_label=d.raw_label, confidence=d.confidence,
                bbox=d.bbox, distance=dist, severity=risk.terms["H"], in_path=in_path, risk=risk))
        frame_risk = self.risk_engine.fuse_frame([r.risk for r in records])
        t_risk = (time.perf_counter() - t0) * 1000.0

        # 6) visualisation
        annotated = frame.copy()
        if draw_path_corridor:
            draw_corridor(annotated, cfg.CORRIDOR_WIDTH_RATIO, self.distance_estimator.camera.horizon_ratio)
        for r in records:
            tag = "sim" if r.distance.is_simulated else "est"
            txt = (f"{r.category} {r.confidence:.2f} | {r.distance.distance_m:.0f}m({tag}) "
                   f"| R={r.risk.score:.2f} {r.risk.level}")
            draw_hazard(annotated, r.bbox, txt, r.risk.level)
        draw_banner(annotated, frame_risk.level, frame_risk.action, frame_risk.score,
                    f"Speed {speed.kmh:.0f} km/h [{'SIMULATED' if speed.is_simulated else 'EST.'}]")

        timings = {
            "preprocess_ms": t_pre,
            "inference_ms": t_det,
            "risk_ms": t_risk,
            "total_ms": (time.perf_counter() - t_start) * 1000.0,
        }
        return FrameResult(annotated, records, frame_risk, speed, timings)