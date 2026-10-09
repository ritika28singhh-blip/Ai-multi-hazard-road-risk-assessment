"""
app.py
======
Professional Streamlit Dashboard for the CAEV-II Mini-Project.
"""
from __future__ import annotations
import time
import tempfile
import os
import streamlit as st
import cv2
import numpy as np

import config as cfg
from detector import HazardDetector
from pipeline import RiskPipeline
from risk_engine import RiskConfig
from speed_estimator import SpeedEstimator
from utils import VideoSource, bgr_to_rgb, FpsMeter

st.set_page_config(
    page_title="CAEV-II Road Risk Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_detector():
    return HazardDetector()

def main():
    st.title("🚗 AI-Based Multi-Hazard Road Risk Assessment System")
    st.markdown("**Connected and Autonomous Vehicles (CAV) Perception & Risk Fusion Prototype**")
    st.markdown("---")

    detector = load_detector()
    pipeline = RiskPipeline(detector)

    # Sidebar Controls
    st.sidebar.header("🎛️ Prototype Configuration")
    
    source_type = st.sidebar.selectbox("Select Video Source", ["Sample Video / Upload", "Webcam (Live)"])
    uploaded_file = None
    webcam_index = 0

    if source_type == "Sample Video / Upload":
        uploaded_file = st.sidebar.file_uploader("Upload Road Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])
    else:
        webcam_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=2, value=0, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚗 Vehicle & Sensor Parameters")
    simulated_speed = st.sidebar.slider("Simulated Vehicle Speed (km/h)", min_value=0.0, max_value=120.0, value=45.0, step=5.0)
    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.1, max_value=0.9, value=cfg.DEFAULT_CONF_THRESHOLD, step=0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Risk Fusion Weights (w1 - w4)")
    w1 = st.sidebar.slider("w1 (Confidence C)", 0.0, 1.0, cfg.DEFAULT_WEIGHTS["w1"], 0.05)
    w2 = st.sidebar.slider("w2 (Inverse Distance 1/D)", 0.0, 1.0, cfg.DEFAULT_WEIGHTS["w2"], 0.05)
    w3 = st.sidebar.slider("w3 (Speed S)", 0.0, 1.0, cfg.DEFAULT_WEIGHTS["w3"], 0.05)
    w4 = st.sidebar.slider("w4 (Hazard Severity H)", 0.0, 1.0, cfg.DEFAULT_WEIGHTS["w4"], 0.05)
    
    # Update pipeline risk config
    risk_cfg = RiskConfig(weights={"w1": w1, "w2": w2, "w3": w3, "w4": w4})
    pipeline.set_risk_config(risk_cfg)

    speed_estimator = SpeedEstimator()

    # Main Layout Metrics
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📹 Real-Time Hazard Detection & Risk Feed")
        video_container = st.empty()

    with col2:
        st.subheader("⚠️ Threat Status")
        risk_metric_placeholder = st.empty()
        action_metric_placeholder = st.empty()

    with col3:
        st.subheader("📊 Performance Stats")
        stats_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📋 Active Hazards in Current Frame")
    hazards_table_placeholder = st.empty()

    run_button = st.sidebar.button("🚀 Run Assessment Pipeline", type="primary")

    if run_button or source_type == "Webcam (Live)":
        target_source = 0 if source_type == "Webcam (Live)" else (uploaded_file.name if uploaded_file else None)
        
        if source_type == "Sample Video / Upload" and uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            target_source = tfile.name

        if target_source is not None:
            fps_meter = FpsMeter()
            vid_source = VideoSource(target_source)

            if not vid_source.is_opened():
                st.error("❌ Failed to open video source. Please check the file or webcam connection.")
                return

            stop_feed = st.sidebar.button("⏹️ Stop Stream")

            while vid_source.is_opened() and not stop_feed:
                ok, frame = vid_source.read()
                if not ok:
                    break

                current_speed = speed_estimator.get_speed(simulated_speed)
                result = pipeline.process(frame, current_speed, conf=conf_threshold, draw_path_corridor=True)
                current_fps = fps_meter.tick()

                # Render Video Frame
                rgb_frame = bgr_to_rgb(result.annotated_bgr)
                video_container.image(rgb_frame, channels="RGB", use_column_width=True)

                # Render Risk Metrics
                fr = result.frame_risk
                level_color = "green" if fr.level == "LOW" else ("orange" if fr.level == "MEDIUM" else "red")
                risk_metric_placeholder.markdown(f"### Risk Level: :{level_color}[**{fr.level}**] (Score: `{fr.score:.2f}`)")
                action_metric_placeholder.markdown(f"### Action: **{fr.action}**")

                # Render Stats
                stats_placeholder.markdown(f"""
                - **Detected Hazards:** `{len(result.hazards)}`
                - **Inference Latency:** `{result.timings_ms.get('inference_ms', 0):.1f} ms`
                - **Measured FPS:** `{current_fps:.1f}`
                - **Vehicle Speed:** `{current_speed.kmh:.0f} km/h`
                """)

                # Render Hazard Table
                if result.hazards:
                    table_data = []
                    for h in result.hazards:
                        table_data.append({
                            "Category": h.category,
                            "Confidence": f"{h.confidence:.2f}",
                            "Distance (m)": f"{h.distance.distance_m}m",
                            "Severity (H)": f"{h.severity:.2f}",
                            "Risk Score": f"{h.risk.score:.2f}",
                            "Level": h.risk.level,
                            "Action": h.risk.action
                        })
                    hazards_table_placeholder.dataframe(table_data, use_container_width=True)
                else:
                    hazards_table_placeholder.info("No hazards detected in the current frame.")

            vid_source.release()
        else:
            st.warning("⚠️ Please upload a valid sample video or select Webcam source.")

if __name__ == "__main__":
    main()