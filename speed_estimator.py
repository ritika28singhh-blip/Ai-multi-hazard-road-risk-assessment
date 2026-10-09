"""
speed_estimator.py
================  
Handles vehicle speed inputs (supports user configuration / simulation).
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SpeedReading:
    kmh: float
    is_simulated: bool = True

class SpeedEstimator:
    def __init__(self, default_speed_kmh: float = 40.0):
        self.default_speed = default_speed_kmh

    def get_speed(self, simulated_kmh: float) -> SpeedReading:
        return SpeedReading(kmh=float(simulated_kmh), is_simulated=True)