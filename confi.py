"""
config.py
=========
Global configuration constants, risk weights, severity values, and class mappings.
"""
from __future__ import annotations
from pathlib import Path
import cv2

# Directory Paths
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Processing Constraints
PROCESS_MAX_WIDTH = 640
DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_GENERAL_MODEL = "yolov8n.pt"
POTHOLE_MODEL_PATH = MODELS_DIR / "pothole.pt"
DEVICE = None  # None = auto (CPU / CUDA)

# Hazard Categories
CATEGORIES = ["Pothole", "Pedestrian", "Vehicle", "Obstruction", "Traffic Sign/Signal"]

# Hazard Severity / Category Factor (H)
HAZARD_SEVERITY = {
    "Pedestrian": 1.00,
    "Vehicle": 0.85,
    "Pothole": 0.70,
    "Obstruction": 0.65,
    "Traffic Sign/Signal": 0.40,
}

# COCO Class to Category Mapping
COCO_TO_CATEGORY = {
    "person": "Pedestrian",
    "bicycle": "Vehicle",
    "car": "Vehicle",
    "motorcycle": "Vehicle",
    "airplane": "Vehicle",
    "bus": "Vehicle",
    "train": "Vehicle",
    "truck": "Vehicle",
    "boat": "Vehicle",
    "traffic light": "Traffic Sign/Signal",
    "stop sign": "Traffic Sign/Signal",
    "fire hydrant": "Obstruction",
    "stop sign": "Traffic Sign/Signal",
    "bench": "Obstruction",
    "dog": "Obstruction",
    "cat": "Obstruction",
    "horse": "Obstruction",
    "cow": "Obstruction",
}

KEYWORD_TO_CATEGORY = [
    (["pothole", "hole", "patch", "crack"], "Pothole"),
    (["debris", "obstacle", "stone", "rock", "animal"], "Obstruction"),
    (["sign", "signal", "light"], "Traffic Sign/Signal"),
    (["car", "truck", "bus", "bike", "vehicle"], "Vehicle"),
    (["person", "pedestrian", "people", "walker"], "Pedestrian"),
]

# Risk Formulation Weights: R = w1*C + w2*(1/D) + w3*S + w4*H
DEFAULT_WEIGHTS = {
    "w1": 0.25,  # Detection Confidence (C)
    "w2": 0.30,  # Inverse Distance factor (1/D)
    "w3": 0.20,  # Normalized Speed factor (S)
    "w4": 0.25,  # Hazard Severity factor (H)
}

# Risk Thresholds
RISK_THRESHOLDS = {
    "MEDIUM": 0.45,  # Score >= 0.45 -> MEDIUM
    "HIGH": 0.70,    # Score >= 0.70 -> HIGH
}

# Risk Actions Mapping
RISK_ACTIONS = {
    "LOW": "MONITOR",
    "MEDIUM": "CAUTION",
    "HIGH": "WARNING",
}

# UI Visualization Colors (BGR format for OpenCV)
LEVEL_COLORS_BGR = {
    "LOW": (0, 200, 0),       # Green
    "MEDIUM": (0, 165, 255),  # Orange
    "HIGH": (0, 0, 255),      # Red
}

CORRIDOR_WIDTH_RATIO = 0.45   # Central driving path corridor ratio