# Emotitron

**Real-time biometric and emotion intelligence platform powered by open-source AI models.**

Emotitron streams live webcam frames through a four-stage ONNX inference pipeline — face detection, anti-spoofing, identity matching, and emotion recognition — and returns structured per-frame analytics over a WebSocket connection. Authenticated users can enroll a biometric face template, run live inference, persist emotion records, and explore historical affective data through an interactive React dashboard.

<p align="center">
  <img src="./frontend/src/assets/emotitrackincon.jpg" alt="Web App Main Icon" width="256">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [WebSocket Stream](#websocket-stream)
- [Face Geometry](#face-geometry)
- [Database Schema](#database-schema)
- [Frontend](#frontend)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Overview

Emotitron is a full-stack web application that performs real-time biometric and affective analysis through a browser webcam. Each video frame is sent to a FastAPI backend where four ONNX models run sequentially:

1. **Face detection** — locates the face and extracts 5 landmarks
2. **Liveness / anti-spoofing** — rejects photos and screen replays
3. **Biometric matching** — compares a live 512D ArcFace embedding against the enrolled template
4. **Emotion recognition** — classifies 8 emotion categories with probability scores

Results are returned as a JSON payload per frame and rendered live in the UI. Emotion events can be saved to PostgreSQL and explored through charts, timelines, Russell circumflex space, and 3D PCA embedding scatter plots.

**Key capabilities:**
- Real-time webcam inference at ~3 FPS over WebSocket
- Biometric enrollment with liveness-gated face template storage
- Per-frame geometry metrics: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head pose (pitch/yaw/roll), and behavioral expression signals
- Shannon entropy and Russell valence/arousal mapping for each emotion result
- Paginated emotion history with date and class filters
- PCA dimensionality reduction of 512D face embeddings, visualized in both 2D (SVG) and 3D (Three.js)
- Biometric-gated profile updates (sensitive changes require live face verification)
- JWT authentication with OAuth2 password flow

---

## Tech Stack

### Backend

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI 0.110.0 (Python 3.12) |
| Server | Uvicorn with `--reload` in development |
| Database | PostgreSQL + pgvector extension |
| ORM | SQLAlchemy 2.0.35 |
| Migrations | Alembic |
| ML Inference | ONNX Runtime 1.20.0 |
| Computer Vision | OpenCV 4.10 (headless) + NumPy 1.26.4 |
| Auth | JWT via python-jose, passwords via passlib + bcrypt 4.0.1 |

### Frontend

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.2.0 | UI framework |
| Vite | 5.0.8 | Build tool and dev server |
| React Router | 6.22.0 | Client-side routing |
| Tailwind CSS | 3.4.19 | Utility-first styling |
| Axios | 1.6.7 | HTTP client |
| Recharts | 3.8.0 | 2D charts (radar, line, bar, pie) |
| Three.js | 0.183.2 | 3D rendering engine |
| @react-three/fiber | 8.18.0 | React wrapper for Three.js |
| @react-three/drei | 9.122.0 | Three.js helpers (OrbitControls, etc.) |

Visual theme: **Purple Cyberpunk** — dark backgrounds, neon purple/green accents, monospace terminal aesthetics.

### Infrastructure

- Docker Compose with three services: `emotrack_db`, `emotrack_backend`, `emotrack_frontend`
- CI via GitHub Actions (`CI_Test.yml`, `PR_Checks.yml`) — runs tests against SQLite in-memory

---

## ML Pipeline

ONNX model files are located in `backend/ml_weights/` and are loaded once at FastAPI startup via a lifespan context manager.

```
backend/ml_weights/
├── detection/det_500m.onnx          SCRFD face detection + 5 landmarks
├── liveness/minifasnet_v2.onnx      MiniFASNetV2 anti-spoofing
├── recognition/w600k_mbf.onnx       ArcFace 512D face embeddings
└── emotion/emotieff_b0.onnx         EmotiEffLib EfficientNet-B0 emotion classification
```

### Per-Frame Processing Order

Each WebSocket frame passes through stages sequentially. A stage only runs if all prior stages succeeded.

```
Frame received
    │
    ├─ Motion detection (pre-filter)
    │   160×120 grayscale diff — skip if mean diff < 2.0
    │
    ├─ [1] Face Detection — SCRFD (det_500m.onnx)
    │       Input:  BGR image → resized 640×640, normalized (pixel−127.5)/128
    │       Output: bounding boxes + 5-point landmarks (L-eye, R-eye, nose, L-mouth, R-mouth)
    │       Stop if no face detected
    │
    ├─ [2] Liveness — MiniFASNetV2 (minifasnet_v2.onnx)
    │       Input:  face crop from bounding box
    │       Output: liveness_score (0–1)
    │       Composite gate: model_score > 0.65 AND EAR > 0.125
    │       Biometric and emotion stages skipped if SPOOF
    │
    ├─ [3] Face Geometry (pure NumPy — no ONNX)
    │       EAR  → eye state (open / blinking / drowsy)
    │       MAR  → mouth state (open / yawning)
    │       Head pose → yaw / pitch / roll via SolvePnP + Rodrigues
    │       Expressions → smile, talking, attention (geometric heuristics)
    │
    ├─ [4] Biometric Matching — ArcFace (w600k_mbf.onnx)
    │       Input:  aligned face 112×112 RGB, affine-transformed from landmarks
    │       Output: 512D L2-normalized embedding
    │       Match:  cosine similarity vs enrolled template (threshold 0.5)
    │       Only runs if user has enrolled face_embedding
    │
    └─ [5] Emotion — EmotiEffLib (emotieff_b0.onnx)
            Input:  aligned face 112×112 RGB
            Output: 8-class probabilities
            Classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
            Derived: Shannon entropy, Russell (valence, arousal) coordinates
```

### Inference Thresholds

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Face detection score | 0.3 | Minimum SCRFD confidence |
| Liveness score | 0.65 | Minimum to pass anti-spoof gate |
| Biometric match | 0.5 | Cosine similarity threshold |
| EAR blink | 0.22 | Eye Aspect Ratio — blinking |
| EAR drowsy | 0.18 | EAR sustained 15+ frames — drowsy |
| MAR yawn | 0.60 | Mouth Aspect Ratio — yawning |
| Frontal pose | ±15° yaw/pitch | Head pose within frontal range |

---

## Architecture

```
Browser (React + Vite)
    │
    │  WebSocket — binary JPEG frames, ~3 FPS
    │  REST — auth, profile, emotions, analytics
    ▼
FastAPI Backend (Docker — emotrack_backend:8000)
    │
    ├── ML Pipeline (ONNX Runtime, in-memory)
    │       SCRFD → MiniFASNetV2 → ArcFace → EmotiEffLib
    │
    └── PostgreSQL + pgvector (Docker — emotrack_db:5432)
            users  (Vector 512D face_embedding)
            emotions  (JSONB emotion_scores, entropy)
            face_session_embeddings  (Vector 512D, session_id)
```

### Docker Compose Services

| Service | Container | Port | Image |
|---------|-----------|------|-------|
| PostgreSQL + pgvector | `emotrack_db` | 5432 | `ankane/pgvector` |
| FastAPI + ONNX | `emotrack_backend` | 8000 | `./backend/Dockerfile` |
| React + Vite | `emotrack_frontend` | 3000 | `./frontend/Dockerfile` |

Startup order: `db` (health-checked) → `backend` → `frontend`.

---

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- A `.env` file at the project root (see [Environment Variables](#environment-variables))
- ONNX model files placed in `backend/ml_weights/` (see `backend/ml_weights/get_models.py`)

### Run with Docker Compose

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

### Run Backend Locally (without Docker)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Requires a running PostgreSQL instance with the pgvector extension enabled, and all environment variables set.

### Run Frontend Locally

```bash
cd frontend
npm install
npm run dev       # Dev server at localhost:3000
npm run build     # Production build
npm run preview   # Preview production build
```

### Database Migrations

```bash
# Inside the running container
docker exec emotrack_backend alembic upgrade head

# Generate a new migration
docker exec emotrack_backend alembic revision --autogenerate -m "description"
```

---

## Environment Variables

The backend reads configuration via `pydantic-settings`. Create a `.env` file at the project root:

```env
POSTGRES_USER=beauAdmin
POSTGRES_PASSWORD=your_password
POSTGRES_DB=emotrack
DB_HOST=db
DB_PORT=5432
SECRET_KEY=your_secret_key_here
ENVIRONMENT=development
```

`DATABASE_URL` is computed automatically from the above. Override with `DATABASE_URL_OVERRIDE` if needed.

---

## API Reference

All REST endpoints are under `/api/v1`. Protected endpoints require `Authorization: Bearer <token>`.

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service banner and docs path |
| GET | `/api/v1/health` | DB connectivity check — returns `{status, database, environment}` |

### Authentication — `/api/v1/auth`

#### `POST /api/v1/auth/register`

Creates a new user account.

**Request body:**
```json
{
  "full_name": "string (2–100 chars)",
  "email": "user@example.com",
  "password": "string (min 8 chars)",
  "age": 25,
  "phone_number": "+1234567890",
  "gender": "M or F",
  "country": "string (optional)"
}
```

**Response:** `UserResponse` — user profile without password hash or face embedding.

#### `POST /api/v1/auth/login`

**Request** (`OAuth2PasswordRequestForm`): `username` = email, `password`.

**Response:**
```json
{ "access_token": "<JWT>", "token_type": "bearer" }
```

---

### Users — `/api/v1/users` _(auth required)_

| Method | Path | Description |
|--------|------|-------------|
| GET | `/me` | Get current user profile |
| PUT | `/me` | Update full_name, email, phone_number, or password |
| DELETE | `/me` | Permanently delete account and all associated data |
| POST | `/me/biometrics` | Enroll face biometric template |
| DELETE | `/me/face_embedding` | Remove stored biometric template |
| POST | `/me/verify` | Verify identity against stored template |

**Biometric enrollment pipeline** (`POST /me/biometrics`):
1. Validate image format
2. Decode via OpenCV
3. SCRFD — detect exactly one face
4. MiniFASNetV2 — liveness score must be > 0.65
5. Affine face alignment → ArcFace embedding (512D)
6. Store L2-normalized vector in `users.face_embedding` (pgvector)

**`UserResponse` schema:**
```json
{
  "id": 1,
  "full_name": "string",
  "email": "user@example.com",
  "age": 25,
  "gender": "M",
  "country": "string",
  "phone_number": "string",
  "is_active": true,
  "has_embedding": false,
  "created_at": "2026-03-01T00:00:00Z",
  "updated_at": "2026-03-01T00:00:00Z"
}
```

`has_embedding` is a derived boolean. The raw 512D vector is never exposed in any response.

---

### Emotions — `/api/v1/emotions` _(auth required)_

Emotions are **never auto-saved** by the WebSocket stream. Saving requires an explicit `POST /save` from the frontend.

Valid emotion classes: `Anger`, `Contempt`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/history` | Paginated records with optional `emotion_filter`, `date_from`, `date_to` |
| GET | `/summary` | Aggregated counts, percentages, and average confidence per class |
| GET | `/details?emotion=Happiness` | Class-level breakdown for one specific emotion |
| GET | `/scores?limit=10` | Last N records with full probability distributions and entropy |
| GET | `/scores/chart` | Histogram-style aggregated distribution across emotion classes |
| GET | `/timeline?limit=100` | Chronological records oldest-first for time-series charts |
| POST | `/save` | Save an emotion record; computes and stores Shannon entropy automatically |

**`POST /save` request body:**
```json
{
  "dominant_emotion": "Happiness",
  "confidence": 0.91,
  "emotion_scores": { "Happiness": 0.91, "Neutral": 0.06, "...": 0.0 },
  "user_note": "optional string (max 500 chars)"
}
```

---

### Analytics — `/api/v1/analytics` _(auth required)_

| Method | Path | Description |
|--------|------|-------------|
| GET | `/pca` | Compute 3D PCA from registered + session embeddings |
| POST | `/session/embed` | Store a 512D session embedding for PCA |
| GET | `/session/history?limit=50` | Recent session embedding metadata (no raw vectors) |
| DELETE | `/session` | Delete all session embeddings for the current user |

**`GET /pca` response:**
```json
{
  "points": [
    { "x": -0.42, "y": 0.18, "z": -0.07, "user_id": 1, "source": "registered", "label": "John Doe", "is_current_user": true },
    { "x": 0.31, "y": -0.22, "z": 0.14, "user_id": 1, "source": "session", "label": "session_abc123", "is_current_user": true }
  ],
  "explained_variance": [0.41, 0.22, 0.11],
  "cumulative_variance": [0.41, 0.63, 0.74],
  "total_variance": 0.74,
  "total_points": 45,
  "registered_count": 12,
  "session_count": 33
}
```

Requires a minimum of 3 samples. Returns `{"error": "..."}` with 200 if insufficient data.

---

## WebSocket Stream

**Endpoint:** `ws://localhost:8000/ws/stream?token=<JWT>`

The endpoint accepts two frame formats:

| Format | Description |
|--------|-------------|
| Binary bytes | Raw JPEG — `websocket.send(blob)` from canvas |
| JSON text | `{"image": "data:image/jpeg;base64,..."}` |

### Full Response Payload

```json
{
  "status": "success | no_face_detected | skipped_low_motion",

  "bbox": [x1, y1, x2, y2],

  "liveness": {
    "is_live": true,
    "score": 0.92,
    "texture_score": 145.3
  },

  "biometrics": {
    "is_match": true,
    "similarity_score": 0.87
  },

  "emotion": {
    "dominant_emotion": "Happiness",
    "confidence": 0.91,
    "emotion_scores": {
      "Anger": 0.01, "Contempt": 0.01, "Disgust": 0.01,
      "Fear": 0.01, "Happiness": 0.91, "Neutral": 0.04,
      "Sadness": 0.00, "Surprise": 0.01
    },
    "entropy": 0.42
  },

  "geometry": {
    "ear": { "ear": 0.28, "eye_state": "open", "is_blinking": false, "is_drowsy": false },
    "mar": { "mar": 0.12, "is_yawning": false },
    "head_pose": {
      "pitch": -2.1, "yaw": 3.4, "roll": 0.8,
      "pose_label": "frontal", "is_frontal": true
    },
    "expressions": {
      "is_smiling": true, "is_duchenne_smile": false, "is_talking": false,
      "smile_score": 0.74, "talk_score": 0.05, "engagement_score": 0.78
    },
    "attention": { "is_distracted": false, "attention_state": "focused" }
  },

  "metrics": {
    "face_detection_ms": 18.4,
    "liveness_ms": 6.2,
    "embedding_ms": 8.1,
    "emotion_ms": 11.3
  },

  "ml_pipeline": {
    "face_detected": true,
    "liveness": "LIVE",
    "biometric_match": "MATCH | NO_MATCH | NOT_AVAILABLE | NOT_RUN",
    "emotion": "Happiness"
  },

  "analytics": {
    "timestamp": 1741382400.0,
    "ear": 0.28, "mar": 0.12,
    "yaw": 3.4, "pitch": -2.1,
    "smile_score": 0.74, "engagement_score": 0.78
  }
}
```

Frame interval: **300 ms** (~3 FPS sent). Frontend captures at 320×240, JPEG quality 0.5.

---

## Face Geometry

Computed in `backend/app/services/face_geometry.py` from the 5-point SCRFD landmark set — no additional ONNX model required.

**Landmark indices:** 0 = Left Eye, 1 = Right Eye, 2 = Nose, 3 = Mouth Left, 4 = Mouth Right

### Eye Aspect Ratio (EAR)

Approximated from the interocular midline-to-nose distance as a vertical proxy (SCRFD provides center points, not full eyelid contours).

| EAR Value | Classification |
|-----------|---------------|
| < 0.22 | Blinking / closed |
| 0.22 – 0.25 | Closing |
| 0.25 – 0.35 | Normal open range |
| < 0.18 for 15+ frames | Drowsy |

### Mouth Aspect Ratio (MAR)

| MAR Value | Classification |
|-----------|---------------|
| > 0.60 | Yawning |

### Head Pose

Estimated via the Perspective-n-Point (PnP) algorithm (`cv2.solvePnP`) using a generic 3D facial model with direct Euler angle extraction from the rotation matrix.

| Axis | Threshold for Frontal |
|------|-----------------------|
| Yaw (left/right) | ±15° |
| Pitch (up/down) | ±15° |
| Roll (tilt) | ±20° |

### Expression Scoring Heuristics

| Signal | Method |
|--------|--------|
| `smile_score` | `mouth_width / interocular_distance > 0.55` AND `MAR < 0.40` |
| `is_duchenne_smile` | Genuine smile AND `EAR < 0.30` (eyes narrowing) |
| `talk_score` | MAR variance over a 6-frame temporal window, threshold 0.04 |
| `engagement_score` | `attention_factor × happy_score`, 0 if distracted |

All geometry functions implement zero-division guards and graceful degradation when `solvePnP` fails to converge.

---

## Database Schema

```
┌──────────────────────────────────────────────┐
│ users                                         │
│  id PK · email (unique) · full_name          │
│  hashed_password · age · gender · country    │
│  phone_number · is_active · is_superuser     │
│  face_embedding  Vector(512)  ← pgvector     │
│  created_at · updated_at                     │
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

pgvector stores 512D ArcFace vectors natively, enabling cosine similarity operations directly in PostgreSQL.

### Alembic Migration History

| Revision | Change |
|----------|--------|
| `a63ab6c8f35d` | Create `users` table + pgvector extension + `face_embedding Vector(512)` |
| `bb9779230ae7` | Create `emotions` table |
| `8f225089f7e7` | Add `is_superuser` to `users` |
| `ea31fedf1b49` | Create `face_session_embeddings` table |
| `a3c0624aa624` | Add `entropy` column to `emotions` |
| `37fb2ac24236` | Add `country` column to `users` |

---

## Frontend

### Pages

| Route | Page | Auth | Description |
|-------|------|:----:|-------------|
| `/about` | About.jsx | No | Landing page |
| `/login` | Login.jsx | No | Authentication |
| `/register` | Register.jsx | No | Account creation |
| `/dashboard` | Dashboard.jsx | Yes | Overview summary |
| `/inference` | Inference.jsx | Yes | Live webcam inference |
| `/emotions` | Emotions.jsx | Yes | Emotion dashboard with charts |
| `/report` | EmotionReport.jsx | Yes | Detailed multi-panel analysis |
| `/russelquadrants` | EmotionsAnalysis.jsx | Yes | Russell circumflex space |
| `/history` | History.jsx | Yes | Paginated emotion history |
| `/analytics` | Analytics.jsx | Yes | 2D PCA scatter plot (SVG) |
| `/pcaanalytics` | PCAAnalytics.jsx | Yes | 3D PCA scatter (Three.js) |
| `/profile` | Profile.jsx | Yes | User settings + biometrics |

### Live Inference Page (`/inference`)

The primary real-time face analysis view renders:
- Live video feed with bounding box overlay
- `StreamMetricsHUD` — ML timing overlay (`face_detection_ms`, `liveness_ms`, `embedding_ms`, `emotion_ms`)
- `ExpressionSignalsHUD` — smile score, talk score, attention state
- EAR / MAR meters
- Liveness badge (LIVE / SPOOF / UNKNOWN)
- Biometric match indicator
- Dominant emotion + confidence display
- FPS / Latency / Throughput metrics
- **SAVE EMOTION** button → `POST /emotions/save`
- **CAPTURE EMBEDDING** button → `POST /analytics/session/embed`

### PCA Visualization

Two separate views for exploring face embedding space:

- **`/analytics`** — 2D SVG scatter (PC1 vs PC2) with color-coded points: current user (neon green), registered users (purple), session captures (faded). Hover tooltip shows PC1/PC2/PC3 values.
- **`/pcaanalytics`** — Interactive Three.js 3D point cloud with OrbitControls (orbit, zoom, pan). Variance spectrum panel and legend overlay.

### Affective Computing Utilities

The frontend includes a set of signal-processing utilities for temporal emotion analysis:

| Module | Capabilities |
|--------|-------------|
| `utils/russellMapping.js` | Maps 8-class probabilities to Russell 2D circumflex (valence × arousal) |
| `core/affective/emotionMetrics.js` | Energy, momentum, drift, turbulence, stability, regime classification |
| `core/affective/emotionDynamics.js` | Velocity, acceleration, angular velocity in Russell space |

### `useFaceTracking` Hook

Central hook managing the WebSocket connection and frame loop:

- Opens `getUserMedia` at 640×480, 10 FPS
- Draws to a 320×240 canvas, encodes as JPEG at quality 0.5
- Sends binary blob every 300 ms
- Tracks EMA-smoothed latency (α=0.2) and throughput (α=0.25)
- Debounces UI state updates at 150 ms to prevent excessive re-renders
- Exposes: `results`, `isConnected`, `error`, `fps`, `latency`, `throughput`, `emotionScores`, `events`

---

## Testing

All tests live in `backend/tests/` and run against **SQLite in-memory**, not PostgreSQL.

```bash
# Run all tests via Docker
docker exec emotrack_backend pytest

# Run specific file with verbose output
docker exec emotrack_backend pytest tests/test_integration_auth.py -v -s

# Run unit tests only
docker exec emotrack_backend pytest tests/test_face_math.py tests/test_face_geometry.py tests/test_inference_engine.py

# Run integration tests only
docker exec emotrack_backend pytest tests/test_integration_auth.py tests/test_integration_users.py tests/test_integration_emotions.py
```

### Test Coverage

| Test File | Coverage Area |
|-----------|--------------|
| `test_face_math.py` | Cosine similarity, PCA reduction, embedding norms |
| `test_face_geometry.py` | EAR, MAR, head pose calculations |
| `test_inference_engine.py` | Model loading and inference via mocked ONNX sessions |
| `test_global.py` | Global behavior and configuration |
| `test_health.py` | Health endpoint behavior |
| `test_integration_auth.py` | Registration, login, JWT flows |
| `test_integration_users.py` | Profile CRUD, biometric enrollment |
| `test_integration_emotions.py` | Emotion save, history, summary, scores |

**Total: 120 test functions, all passing.**

### Critical Test Patterns

**SQLite shared cache** (prevents "no such table" errors when multiple connections touch the same in-memory DB):
```python
SQLITE_URL = "sqlite:///:memory:?cache=shared"
engine = create_engine(SQLITE_URL,
    connect_args={"check_same_thread": False, "uri": True},
    poolclass=StaticPool)
```

**Override both `get_db` imports** (routers import from two different modules):
```python
from app.core.session import get_db as session_get_db
from app.core.database import get_db as database_get_db

app.dependency_overrides[session_get_db] = override_get_db
app.dependency_overrides[database_get_db] = override_get_db
```

**Empty lifespan** (skip ONNX model loading and PostgreSQL init during tests):
```python
@asynccontextmanager
async def _empty_lifespan(_app):
    yield
app.router.lifespan_context = _empty_lifespan
```

### Critical Version Pins

```
bcrypt==4.0.1           # bcrypt ≥ 5.0 is incompatible with passlib 1.7.4
passlib[bcrypt]==1.7.4
numpy==1.26.4
onnxruntime==1.20.0
sqlalchemy==2.0.35
pgvector==0.3.6
fastapi==0.110.0
```

---

## Project Structure

```
Emotitron/
├── docker-compose.yml
├── .env                              # PostgreSQL credentials + SECRET_KEY
├── .github/workflows/               # CI_Test.yml, PR_Checks.yml
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic/                     # DB migrations (6 revisions)
│   ├── ml_weights/                  # ONNX model files (not in git)
│   │   ├── detection/det_500m.onnx
│   │   ├── liveness/minifasnet_v2.onnx
│   │   ├── recognition/w600k_mbf.onnx
│   │   └── emotion/emotieff_b0.onnx
│   └── app/
│       ├── main.py                  # FastAPI app + lifespan context
│       ├── api/
│       │   ├── routers/             # auth, users, emotions, analytics, inference
│       │   ├── websockets/          # stream.py, manager.py
│       │   └── dependencies.py     # JWT auth dependencies
│       ├── core/                    # config, database, session, security, logging
│       ├── models/                  # SQLAlchemy ORM models
│       ├── schemas/                 # Pydantic request/response schemas
│       ├── services/                # inference_engine, face_geometry, face_math,
│       │                            #   emotion_math, analytics
│       └── utils/                   # image_processing, image_helper
│
├── tests/
│   ├── conftest.py
│   ├── factories.py
│   ├── mocks.py
│   ├── test_face_geometry.py
│   ├── test_face_math.py
│   ├── test_inference_engine.py
│   ├── test_integration_auth.py
│   ├── test_integration_emotions.py
│   └── test_integration_users.py
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                  # Route definitions + auth guard
        ├── api/                     # Axios API wrappers
        ├── components/              # 28+ reusable UI components
        ├── pages/                   # 12 page components
        ├── hooks/                   # useFaceTracking, usePCAData
        ├── context/                 # AuthContext
        ├── core/affective/          # emotionDynamics, emotionMetrics, emotionFlowField
        ├── utils/                   # russellMapping, emotionDynamics
        ├── config/                  # inference frame configuration
        └── styles/                  # Tailwind + dashboard CSS
```

---

## Recommended Usage Conditions

- Good front lighting, single face in frame
- Frontal head pose for optimal detection and geometry accuracy
- Stable camera with low motion blur
- Avoid presenting screen replays or photos — liveness check will reject spoofed inputs

---

## License

See [LICENSE](./LICENSE) for terms.
