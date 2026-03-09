# Architecture — FaceEmotionTrackAI

## System Overview

FaceEmotionTrackAI is a real-time biometric and affective computing platform. It streams live video frames from a browser webcam to a FastAPI backend, runs a four-stage ML pipeline on each frame, and returns structured JSON back to the React frontend for visualization.

```
Browser (React)
    │
    │  WebSocket (binary JPEG frames, ~3 FPS)
    │  REST (auth, profile, emotions, analytics)
    ▼
FastAPI Backend (Docker)
    │
    ├── ML Pipeline (ONNX Runtime, in-memory)
    │       SCRFD → MiniFASNetV2 → ArcFace → EmotiEffLib
    │
    └── PostgreSQL + pgvector (Docker)
            users (Vector 512D), emotions (JSONB), face_session_embeddings
```

---

## Docker Compose Services

| Service | Container | Port | Image |
|---------|-----------|------|-------|
| PostgreSQL + pgvector | `emotrack_db` | 5432 | `ankane/pgvector` |
| FastAPI + ONNX | `emotrack_backend` | 8000 | Custom `./backend/Dockerfile` |
| React + Vite | `emotrack_frontend` | 3000 | Custom `./frontend/Dockerfile` |

Startup order: `db` (health-checked) → `backend` → `frontend`.

The backend mounts source code as a volume and runs with `--reload`, enabling live development.

---

## ML Pipeline

Each WebSocket frame passes through stages in sequence. A stage only runs if the previous one succeeded.

```
Frame received
    │
    ├─ Motion detection (pre-filter, 160×120 grayscale diff)
    │   └─ Skip if mean diff < 2.0
    │
    ├─ [1] Face Detection — SCRFD (det_500m.onnx)
    │       Input:  BGR image → resized to 640×640, normalized (pixel−127.5)/128
    │       Output: bounding boxes + 5-point landmarks (L-eye, R-eye, nose, L-mouth, R-mouth)
    │       └─ Stop if no face
    │
    ├─ [2] Liveness — MiniFASNetV2 (minifasnet_v2.onnx)
    │       Input:  face crop from bbox
    │       Output: liveness_score (0–1)
    │       Composite: requires model_score > 0.65 AND EAR > 0.125
    │       └─ Biometric + emotion skipped if SPOOF
    │
    ├─ [3] Face Geometry (pure NumPy, no ONNX)
    │       From 5-point landmarks:
    │         EAR  → eye state (open / blinking / drowsy)
    │         MAR  → mouth state (open / yawning)
    │         Head pose → yaw / pitch / roll via SolvePnP + Rodrigues
    │         Expressions → smile, talking, attention (geometric heuristics)
    │
    ├─ [4] Biometric Matching — ArcFace (w600k_mbf.onnx)
    │       Input:  aligned face (112×112 RGB), affine-transformed from landmarks
    │       Output: 512D L2-normalized embedding
    │       Match:  cosine similarity vs stored user embedding (threshold 0.5)
    │       └─ Only runs if user has enrolled face_embedding
    │
    └─ [5] Emotion — EmotiEffLib (emotieff_b0.onnx)
            Input:  aligned face (112×112 RGB)
            Output: 8-class probabilities [Anger, Contempt, Disgust, Fear,
                    Happiness, Neutral, Sadness, Surprise]
            Derived: Shannon entropy, Russell (valence, arousal) coordinates
```

### ONNX Model Files

```
backend/ml_weights/
├── detection/det_500m.onnx          SCRFD face detection + landmarks
├── recognition/w600k_mbf.onnx       ArcFace 512D face embeddings
├── liveness/minifasnet_v2.onnx      MiniFASNetV2 anti-spoofing
└── emotion/emotieff_b0.onnx         EmotiEffLib emotion classification
```

All four models are loaded once at startup via the FastAPI lifespan context manager and held in memory as ONNX `InferenceSession` objects.

---

## Data Flow

### Real-Time Stream (WebSocket)

```
Frontend                            Backend
────────                            ───────
getUserMedia(640×480)
canvas.toBlob(JPEG, 0.5, 320×240)
websocket.send(blob)        ──────► decode_jpeg_bytes()
                                    │
                                    ├─ SCRFD detect_faces()
                                    ├─ analyze_face_geometry()
                                    ├─ check_liveness()
                                    ├─ get_face_embedding()  ──► cosine_similarity()
                                    └─ detect_emotion()      ──► compute_entropy()
                                    │
websocket.onmessage(json)   ◄────── JSON response (bbox, liveness,
                                     biometrics, emotion, geometry,
                                     metrics, analytics)
```

Frame interval: **300 ms** (~3 FPS sent). Motion pre-filter skips frames with mean pixel diff < 2.0 on 160×120 downsampled grayscale.

### Emotion Persistence (REST)

Emotions are **never auto-saved** by the WebSocket stream. The user explicitly clicks "SAVE EMOTION" in the UI, which sends a `POST /api/v1/emotions/save` request with the last inference result.

### PCA Analytics (REST, on-demand)

```
GET /api/v1/analytics/pca
    │
    ├─ fetch_registered_embeddings()  all active users with face_embedding
    ├─ fetch_session_embeddings()     recent FaceSessionEmbedding records
    ├─ stack embeddings (N × 512)
    ├─ SVD → project to 3D
    └─ return points with metadata (source, user_id, is_current_user)
```

---

## Database Schema

```
┌──────────────────────────────────────────────┐
│ users                                         │
│  id PK, email (unique), full_name             │
│  hashed_password, age, gender, country        │
│  phone_number, is_active, is_superuser        │
│  face_embedding  Vector(512)  ← pgvector      │
│  created_at, updated_at                       │
└────────────────┬────────────────┬─────────────┘
                 │                │
    ┌────────────▼───┐   ┌────────▼──────────────────┐
    │ emotions        │   │ face_session_embeddings    │
    │  id PK          │   │  id PK                     │
    │  user_id FK     │   │  user_id FK                │
    │  dominant_      │   │  embedding  Vector(512)    │
    │    emotion      │   │  session_id                │
    │  confidence     │   │  captured_at               │
    │  emotion_scores │   └────────────────────────────┘
    │    JSONB        │
    │  entropy Float  │
    │  timestamp      │
    └─────────────────┘
```

pgvector stores 512D ArcFace vectors natively, enabling cosine similarity queries directly in PostgreSQL.

---

## Authentication

JWT-based. Tokens issued at login (`POST /api/v1/auth/login`) and validated on every protected request via the `get_current_active_user` dependency.

WebSocket authentication: token passed as query parameter — `ws://host/ws/stream?token=<JWT>`.

---

## Directory Structure

```
FaceEmotionTrackAI/
├── docker-compose.yml
├── .env                          # PostgreSQL credentials, SECRET_KEY
├── .github/workflows/            # CI_Test.yml, PR_Checks.yml
├── docs/                         # This documentation
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic/                  # DB migrations (6 revisions)
│   ├── ml_weights/               # ONNX model files (not in git)
│   └── app/
│       ├── main.py               # FastAPI app + lifespan
│       ├── api/
│       │   ├── routers/          # auth, users, emotions, analytics, inference
│       │   ├── websockets/       # stream.py, manager.py
│       │   └── dependencies.py   # JWT auth dependencies
│       ├── core/                 # config, database, session, security, logging
│       ├── models/               # SQLAlchemy ORM
│       ├── schemas/              # Pydantic schemas
│       ├── services/             # inference_engine, face_geometry, face_math,
│       │                         #   emotion_math, analytics
│       └── utils/                # image_processing, visual_debug
│
└── frontend/
    ├── package.json
    └── src/
        ├── App.jsx               # Route definitions
        ├── api/                  # axios API wrappers
        ├── components/           # 28+ reusable UI components
        ├── pages/                # 15 page components
        ├── hooks/                # useFaceTracking, usePCAData
        ├── context/              # AuthContext
        ├── core/affective/       # emotionDynamics, emotionMetrics
        ├── utils/                # russellMapping, emotionDynamics
        ├── config/               # inference frame configuration
        └── styles/               # Tailwind + dashboard CSS
```

---

## CI / CD

### `CI_Test.yml`
Triggers on every push to any branch and on PRs targeting `master`.
Runs all unit and integration tests against SQLite in-memory.

### `PR_Checks.yml`
Triggers when a PR targeting `master` is opened, updated, or reopened.
Runs the full test suite and publishes results as a PR comment.

### Test Strategy
Tests use SQLite in-memory (not PostgreSQL). Key implementation patterns:
- `sqlite:///:memory:?cache=shared` + `StaticPool` — all connections share the same in-memory DB
- Empty lifespan override — skips ML model loading and PostgreSQL init
- Both `core/session.get_db` and `core/database.get_db` overridden — different routers import from different modules
- `Vector(512)` → `VectorAsText`, `JSONB` → `JsonAsText` TypeDecorators for SQLite compatibility
