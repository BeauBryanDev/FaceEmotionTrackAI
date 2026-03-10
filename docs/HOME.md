# Welcome to the Emotitron Wiki

**Emotitron** is a cutting-edge, real-time biometric and emotion intelligence platform. By leveraging high-performance open-source AI models, Emotitron transforms live webcam streams into a rich stream of affective and biometric data, providing deep insights into human facial expressions and identity.

---

## 🚀 Overview

Emotitron is designed as a full-stack solution for real-time facial analysis. It streams live video frames from a browser webcam to a powerful FastAPI backend, where a sophisticated four-stage ONNX inference pipeline processes each frame to extract identity, liveness, and emotional state.

### Key Capabilities:
- **Real-Time Inference**: High-speed processing (~3 FPS) over secure WebSockets.
- **Biometric Identity Matching**: Secure 512D ArcFace embedding comparison for identity verification.
- **Liveness Detection**: Integrated anti-spoofing logic to distinguish real human faces from photos or screen replays.
- **Emotion Recognition**: Classification of 8 distinct emotion categories with precise probability scores.
- **Advanced Face Geometry**: Real-time tracking of Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and 3D head pose (Pitch/Yaw/Roll).
- **Interactive Analytics**: Explore affective history through radar charts, timelines, Russell circumflex space, and 3D PCA embedding visualizations.

---

## 🛠️ Technology Stack

Emotitron is built with modern, high-performance technologies:

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12)
- **Database**: [PostgreSQL](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector) for biometric vector storage.
- **ML Engine**: [ONNX Runtime](https://onnxruntime.ai/) for efficient model execution.
- **Frontend**: [React](https://reactjs.org/) + [Vite](https://vitejs.dev/) with [Tailwind CSS](https://tailwindcss.com/) and [Three.js](https://threejs.org/).
- **AI Models**: SCRFD (Detection), MiniFASNetV2 (Liveness), ArcFace (Recognition), and EfficientNet-B0 (Emotion).

---

## 🧬 How It Works

The magic of Emotitron happens in its multi-stage ML pipeline:

1.  **Detection**: Locates the face and 5 key landmarks.
2.  **Liveness**: Validates the subject is a living person.
3.  **Geometry**: Calculates eye/mouth states and head orientation.
4.  **Identity**: Matches the face against a registered biometric template.
5.  **Emotion**: Analyzes expressions to classify the dominant emotional state.

---

## 📚 Wiki Navigation

Explore the detailed documentation to get started or dive deep into the architecture:

- **[Architecture & Design](Architecture)**: Deep dive into the system's structural components.
- **[ML Pipeline Details](ML-Pipeline)**: Technical breakdown of the ONNX models and processing flow.
- **[Getting Started](Installation-Guide)**: Instructions for Docker and local setup.
- **[API Reference](API-Reference)**: Documentation for REST and WebSocket endpoints.
- **[User Guide](User-Guide)**: How to enroll biometrics and use the inference dashboard.
- **[PCA & Analytics](Analytics-Deep-Dive)**: Understanding the embedding visualizations and affective metrics.

---

> "Empowering human-computer interaction through real-time affective intelligence."