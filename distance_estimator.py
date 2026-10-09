"""
distance_estimator.py
=====================
Vision-based distance estimation using bounding-box bottom coordinates
and empirical pinhole camera approximations suitable for laptop software prototypes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class CameraModel:
    focal_length_px: float = 800.0  # Approximate focal length in pixels
    camera_height_m: float = 1.2    # Mounting height of laptop/webcam on dashboard
    horizon_ratio: float = 0.40     # Estimated horizon line vertical ratio

@dataclass
class DistanceEstimate:
    distance_m: float
    is_simulated: bool

class DistanceEstimator:
    def __init__(self, camera: Optional[CameraModel] = None):
        self.camera = camera or CameraModel()

    def estimate(self, bbox: tuple, category: str, raw_label: str, frame_w: int, frame_h: int) -> DistanceEstimate:
        """
        Approximates distance using the vertical position of the bounding box bottom edge (y2).
        Objects closer to the bottom of the frame are closer to the vehicle; objects near the horizon are distant.
        """
        x1, y1, x2, y2 = bbox
        horizon_y = frame_h * self.camera.horizon_ratio
        object_foot_y = float(y2)

        # If bounding box is above or at horizon line, assign a large distance
        if object_foot_y <= horizon_y + 5:
            return DistanceEstimate(distance_m=50.0, is_simulated=False)

        # Pixel height relative to horizon
        pixel_offset = object_foot_y - horizon_y
        
        # Empirical scaling model: distance inversely proportional to pixel offset from horizon
        # D = (focal_constant * camera_height) / pixel_offset
        constant_factor = 350.0  
        distance = constant_factor / max(pixel_offset / frame_h * 100.0, 5.0)
        
        # Clamp realistic bounds for a driving view (2m to 50m)
        distance = max(2.0, min(distance, 50.0))
        return DistanceEstimate(distance_m=round(distance, 1), is_simulated=False)

    def simulated(self, override_distance_m: float) -> DistanceEstimate:
        return DistanceEstimate(distance_m=float(override_distance_m), is_simulated=True)