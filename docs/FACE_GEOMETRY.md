# Face Geometry Service (face_geometry.py)

The Face Geometry Service is a core component of the Computer Vision pipeline. It processes facial landmarks provided by the SCRFD detector to extract physiological metrics (EAR, MAR) and spatial orientation (Head Pose).


## 📌 Overview
This service follows a modular approach to transform raw 2D coordinates into actionable insights for drowsiness and yawn detection. It utilizes an approximate 5-point landmark model to maintain high performance in real-time streaming environments.

### 📈 Performance Metrics
- 📈 Real-time drowsiness detection: EAR < 0.22
- 📈 Yawn detection: MAR > 0.60
- 📈 Head pose estimation: pitch, yaw, roll

### 📈 Input Requirements
- SCRFD keypoints (5x2 array)
- Image width and height

### 📈 Output Format
```python
{
    "ear": {
        "ear": 0.0,
        "eye_state": "unknown",
        "is_blinking": False,
        "is_drowsy": False
    },
    "mar": {
        "mar": 0.0,
        "is_yawning": False
    },
    "head_pose": {
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "pose_label": "unknown",
        "is_frontal": False
    }
}
```

## 🛠 Core Functions
analyze_face_geometry
The primary entry point for the service. It orchestrates all internal calculations and returns a unified object.
Input: landmarks (5x2 array), image_width, image_height, consecutive_frames.
Logic:
Computes Eye Aspect Ratio (EAR) and classifies eye state.
Calculates Mouth Aspect Ratio (MAR).
Estimates 3D Head Pose orientation.
Return: A dictionary optimized for WebSocket transmission and JSON serialization.

compute_ear_from_landmarks
Calculates an approximate Eye Aspect Ratio.
Methodology: Since SCRFD provides 5 points, we use the distance from the inter-ocular midline to the nose as a vertical proxy.
Thresholds:
EAR < 0.22: Classified as a blink or closed eye.
0.25 - 0.35: Normal open-eye range.
estimate_head_pose
Estimates the 3D orientation of the head using the Perspective-n-Point (PnP) algorithm.
Math: Solves the 2D-to-3D correspondence problem using a generic 3D facial model (MODEL_3D_POINTS).
Outputs:
Pitch: Up/Down movement.
Yaw: Left/Right rotation.
Roll: Lateral tilt.
Stability: Uses arctan2 over rotation matrix elements for superior angular stability.


## 📊 Technical Reference: Landmark Indices
The service uses the standard SCRFD 5-point layout:
---
Index	Landmark	Description
0	IDX_LEFT_EYE	Center of the left eye
1	IDX_RIGHT_EYE	Center of the right eye
2	IDX_NOSE	Tip of the nose (Central anchor)
3	IDX_MOUTH_LEFT	Left corner of the mouth
4	IDX_MOUTH_RIGHT	Right corner of the mouth
---

## ⚙️ Configuration Thresholds
These constants are calibrated for standard 720p/1080p webcams:
Blink Threshold: 0.22
Drowsiness Threshold: 0.18 (Sustained for 15+ frames).
Yawn Threshold (MAR): > 0.60.
Frontal Pose: Yaw and Pitch within ±15°.


##  ⚠️ Error Handling & Robustness
The service implements Graceful Degradation:
If solvePnP fails to converge (due to extreme angles or occlusion), the service returns a neutral pose instead of raising an exception.
Zero-division guards are implemented for all ratio calculations (e.g., when landmarks overlap).
All exceptions are caught and logged using app.core.logging.


