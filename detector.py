"""
detector.py
===========
CNN / YOLO-based multi-hazard detection.

Two Ultralytics YOLO models are combined:

* GENERAL model (pretrained on COCO, default `yolov8n.pt`)
    -> pedestrians, vehicles, traffic signs/signals, obstruction-like objects (animals, bench, ...)
* POTHOLE model (custom-trained, `models/pothole.pt`, OPTIONAL)
    -> potholes. COCO has no pothole class, so without this file potholes are NOT detected.

All raw labels are mapped to the five project categories defined in config.py:
    Pothole, Pedestrian, Vehicle, Obstruction, Traffic Sign/Signal

`ultralytics` is imported lazily so that the rest of the project (risk engine, tests)
works without it.
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import config as cfg


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 in the processed frame
    confidence: float
    raw_label: str                    # label as output by the model (e.g. "person")
    category: str                     # one of config.CATEGORIES
    source_model: str                 # "general" or "pothole"


# --------------------------------------------------------------------------- #
# Label -> category mapping
# --------------------------------------------------------------------------- #
def map_label(raw_label: str, default: Optional[str] = None) -> Optional[str]:
    """Map a detector label to a project category (or `default` / None if unknown)."""
    name = raw_label.strip().lower()
    if name in cfg.COCO_TO_CATEGORY:
        return cfg.COCO_TO_CATEGORY[name]
    for keywords, category in cfg.KEYWORD_TO_CATEGORY:
        if any(k in name for k in keywords):
            return category
    return default


@contextlib.contextmanager
def _chdir(path: Path):
    old = os.getcwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_yolo(weights: str | Path):
    """Load YOLO weights. If a bare name like 'yolov8n.pt' is not present in models/,
    Ultralytics downloads it into models/ automatically (internet required once)."""
    from ultralytics import YOLO  # lazy import

    p = Path(weights)
    if p.is_absolute() and p.exists():
        return YOLO(str(p))
    local = cfg.MODELS_DIR / p.name
    if local.exists():
        return YOLO(str(local))
    with _chdir(cfg.MODELS_DIR):          # download target = models/
        model = YOLO(p.name)
    return model


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class HazardDetector:
    def __init__(self, general_weights: str = cfg.DEFAULT_GENERAL_MODEL,
                 pothole_weights: Optional[str | Path] = cfg.POTHOLE_MODEL_PATH,
                 use_pothole_model: bool = True, device: Optional[str] = cfg.DEVICE):
        self.device = device
        self.general_name = str(general_weights)
        self.general = _load_yolo(general_weights)
        self.pothole = None
        self.pothole_name = None
        if use_pothole_model and pothole_weights and Path(pothole_weights).exists():
            self.pothole = _load_yolo(Path(pothole_weights).resolve())
            self.pothole_name = Path(pothole_weights).name

        # only run the COCO classes that map to a project category
        names = self.general.names  # {id: name}
        self._general_class_ids = [i for i, n in names.items() if map_label(n) is not None]

    @property
    def has_pothole_model(self) -> bool:
        return self.pothole is not None

    # ------------------------------------------------------------------------------
    def _run(self, model, frame, conf, iou, imgsz, classes, source_name, default_cat):
        results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, classes=classes,
                                device=self.device, verbose=False)
        out: List[Detection] = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, cf, ci in zip(xyxy, confs, clss):
            label = str(r.names[int(ci)])
            category = map_label(label, default=default_cat)
            if category is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            out.append(Detection((x1, y1, x2, y2), float(cf), label, category, source_name))
        return out

    def detect(self, frame_bgr: np.ndarray, conf: float = cfg.DEFAULT_CONF_THRESHOLD,
               iou: float = cfg.DEFAULT_IOU_THRESHOLD, imgsz: int = cfg.DEFAULT_IMGSZ
               ) -> Tuple[List[Detection], float]:
        """Run detection. Returns (detections, inference_time_ms)."""
        t0 = time.perf_counter()
        dets = self._run(self.general, frame_bgr, conf, iou, imgsz,
                         self._general_class_ids, "general", None)
        if self.pothole is not None:
            dets += self._run(self.pothole, frame_bgr, conf, iou, imgsz, None, "pothole", "Pothole")
        return dets, (time.perf_counter() - t0) * 1000.0