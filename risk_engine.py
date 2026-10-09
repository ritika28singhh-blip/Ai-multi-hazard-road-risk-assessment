"""
risk_engine.py
==============
Implements the exact conceptual risk formulation from the project synopsis:
    R = w1*C + w2*(1/D) + w3*S + w4*H
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import config as cfg

@dataclass
class RiskConfig:
    weights: Dict[str, float] = field(default_factory=lambda: cfg.DEFAULT_WEIGHTS.copy())
    medium_threshold: float = cfg.RISK_THRESHOLDS["MEDIUM"]
    high_threshold: float = cfg.RISK_THRESHOLDS["HIGH"]
    severity_map: Dict[str, float] = field(default_factory=lambda: cfg.HAZARD_SEVERITY.copy())

@dataclass
class RiskResult:
    score: float
    level: str  # LOW, MEDIUM, HIGH
    action: str # MONITOR, CAUTION, WARNING
    terms: Dict[str, float]

@dataclass
class FrameRisk:
    score: float
    level: str
    action: str
    top_index: Optional[int]

class RiskEngine:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def assess(self, category: str, confidence: float, distance_m: float, speed_kmh: float,
               raw_label: str = "", in_path: bool = True,
               speed_is_simulated: bool = True, distance_is_simulated: bool = False) -> RiskResult:
        w = self.config.weights
        
        # Term 1: C = Detection Confidence (0.0 to 1.0)
        C = max(0.0, min(float(confidence), 1.0))

        # Term 2: 1/D = Inverse Distance factor (normalized 0.0 to 1.0)
        # Closer distances yield higher values. Max distance clamped to 50m.
        safe_dist = max(float(distance_m), 1.0)
        inv_d = 1.0 / safe_dist
        D_norm = min(inv_d / (1.0 / 2.0), 1.0)  # normalized against 2m minimum reference

        # Term 3: S = Normalized Vehicle Speed factor (0.0 to 1.0, max ref 120 km/h)
        S = min(max(float(speed_kmh), 0.0) / 120.0, 1.0)

        # Term 4: H = Hazard Severity factor
        H = self.config.severity_map.get(category, 0.5)

        # Apply path relevance penalty if hazard is outside central path corridor
        path_multiplier = 1.0 if in_path else 0.6

        # Exact Risk Formula
        raw_score = (w.get("w1", 0.25) * C +
                     w.get("w2", 0.30) * D_norm +
                     w.get("w3", 0.20) * S +
                     w.get("w4", 0.25) * H) * path_multiplier

        score = round(max(0.0, min(raw_score, 1.0)), 3)

        # Risk Classification
        if score >= self.config.high_threshold:
            level = "HIGH"
        elif score >= self.config.medium_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        action = cfg.RISK_ACTIONS.get(level, "MONITOR")

        return RiskResult(
            score=score,
            level=level,
            action=action,
            terms={"C": round(C, 2), "1/D": round(D_norm, 2), "S": round(S, 2), "H": round(H, 2)}
        )

    def fuse_frame(self, results: List[RiskResult]) -> FrameRisk:
        if not results:
            return FrameRisk(score=0.0, level="LOW", action="MONITOR", top_index=None)
        
        scores = [r.score for r in results]
        max_score = max(scores)
        top_idx = scores.index(max_score)
        top_res = results[top_idx]

        return FrameRisk(
            score=max_score,
            level=top_res.level,
            action=top_res.action,
            top_index=top_idx
        )