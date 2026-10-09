<!-- Animated Banner Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,30,40&height=220&section=header&text=AI%20Multi-Hazard%20Road%20Risk%20Assessment&fontSize=30&animation=fadeIn&fontColor=ffffff&fontAlignY=38&desc=Connected%20and%20Autonomous%20Vehicles%20(CAV)%20Safety%20System&descSize=15&descAlignY=62" width="100%"/>
</p>

<!-- Live Status & Tech Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20%E2%9C%94-success?style=for-the-badge&logo=none" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLO-v8-orange?style=for-the-badge&logo=ultralytics&logoColor=white" alt="YOLO"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge" alt="License"/>
</p>

---

## ⚡ Project Overview
The **AI-Based Multi-Hazard Road Risk Assessment System** is an advanced software prototype engineered for **Connected and Autonomous Vehicles (CAV)**. Built using state-of-the-art computer vision and a mathematical risk-fusion framework, the system processes real-time traffic video feeds to detect roadway obstacles, estimate spatial telemetry, and compute dynamic risk scores to assist vehicular safety logic.

---

## ✨ Key Features

* **👁️ Real-Time Object Detection**: Utilizes **YOLO** and **OpenCV** to track surrounding traffic participants and road elements with high accuracy.
* **📏 Dynamic Distance & Speed Tracking**: Estimates relative proximity and relative velocities from standard monocular video inputs.
* **🧮 Mathematical Risk Fusion Engine**: Integrates multi-factor hazards into a unified safety index using the weighted formula:
  $$\mathcal{R} = w_1 \cdot C + w_2 \cdot \left(\frac{1}{D}\right) + w_3 \cdot S + w_4 \cdot H$$
* **📊 Interactive Web UI**: Powered by **Streamlit** to deliver live video telemetry, instant risk level visualizations, and configurable weight sliders.
* **💻 Zero Hardware Overhead**: Designed to run seamlessly on consumer hardware without requiring expensive physical sensors (LiDAR/CAN bus).

---

## 🏗️ System Architecture & Workflow

```mermaid
graph TD
    A[Video Stream Input] --> B[YOLO Object Detector]
    B --> C[Distance & Speed Estimator]
    C --> D[Risk Fusion Engine]
    D --> E[Streamlit Dashboard UI]
    E --> F[Real-Time Risk Alerts & Telemetry]
