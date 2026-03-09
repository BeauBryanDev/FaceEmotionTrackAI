# Backend — FaceEmotionTrackAI

## Stack

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

---

## Application Entry Point — `app/main.py`

FastAPI app with a lifespan context manager that:
1. Initializes the pgvector extension in PostgreSQL (`init_db()`)
2. Loads all 4 ONNX models into memory (`inference_engine.load_models()`)
3. Clears models on shutdown

CORS is configured to allow all origins. In production this should be restricted.

**Registered routers:**

| Prefix | Module | Purpose |
|--------|--------|---------|
| `/api/v1/auth` | `routers/auth.py` | Registration and login |
| `/api/v1/users` | `routers/users.py` | Profile, biometric enrollment |
| `/api/v1/emotions` | `routers/emotions.py` | Emotion history and analytics |
| `/api/v1/analytics` | `routers/analytics.py` | PCA and session embeddings |
| `/api/v1/inference` | `routers/inference.py` | Single-frame REST inference |
| `/ws/stream` | `websockets/stream.py` | Real-time WebSocket stream |

Health check: `GET /api/v1/health` — queries the database to confirm connectivity.

---

## API Reference

### Authentication — `/api/v1/auth`

#### `POST /api/v1/auth/register`
Creates a new user account.

**Request body** (`UserCreate`):
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

**Response** (`UserResponse`): user profile without `hashed_password` or `face_embedding`.

#### `POST /api/v1/auth/login`
Returns a JWT access token.

**Request** (`OAuth2PasswordRequestForm`): `username` = email, `password`.

**Response**:
```json
{ "access_token": "<JWT>", "token_type": "bearer" }
```

---

### Users — `/api/v1/users`

All endpoints require `Authorization: Bearer <token>`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/me` | Get current user profile |
| PUT | `/me` | Update full_name, email, phone_number, or password |
| DELETE | `/me` | Permanently delete account and all biometric data |
| POST | `/me/face_embedding` | (internal) Update face embedding record |
| DELETE | `/me/face_embedding` | Remove stored biometric template |
| POST | `/me/biometrics` | Enroll face (upload image → SCRFD → liveness → ArcFace → store) |
| POST | `/me/verify` | Verify identity via live face image vs stored template |

**Biometric enrollment pipeline** (`POST /me/biometrics`):
1. Validate image format
2. Decode via OpenCV
3. SCRFD face detection (exactly 1 face required)
4. MiniFASNetV2 liveness check (score > 0.65)
5. Affine face alignment → ArcFace embedding (512D)
6. Store vector in `users.face_embedding` (pgvector)

**`UserResponse` schema**:
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

`has_embedding` is a derived boolean — `true` when `face_embedding` is not null. The raw 512D vector is never exposed in any response.

---

### Emotions — `/api/v1/emotions`

All endpoints require authentication. Emotions are **never auto-saved** by the WebSocket stream — saving requires an explicit `POST /save` call from the frontend.

Valid emotion classes: `Anger`, `Contempt`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`.

#### `GET /api/v1/emotions/history`
Paginated emotion records with optional filters.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `page_size` | int (1–100) | 10 | Records per page |
| `emotion_filter` | string | — | Filter to one emotion class |
| `date_from` | ISO datetime | — | Start of date range |
| `date_to` | ISO datetime | — | End of date range |

**Response**:
```json
{
  "user_id": 1,
  "page": 1,
  "page_size": 10,
  "total_records": 42,
  "total_pages": 5,
  "records": [
    {
      "id": 1,
      "dominant_emotion": "Happiness",
      "confidence": 0.91,
      "emotion_scores": { "Happiness": 0.91, "Neutral": 0.06, ... },
      "entropy": 0.42,
      "timestamp": "2026-03-07T14:23:00Z"
    }
  ]
}
```

#### `GET /api/v1/emotions/summary`
Aggregated statistics across all emotion records.

**Response**:
```json
{
  "user_id": 1,
  "total_detections": 120,
  "dominant_emotion": "Happiness",
  "emotion_stats": [
    { "emotion": "Happiness", "count": 60, "percentage": 50.0, "avg_confidence": 0.88 }
  ]
}
```

#### `GET /api/v1/emotions/details?emotion=Happiness`
Class-level breakdown for one specific emotion.

#### `GET /api/v1/emotions/scores?limit=10`
Last N records with full probability distributions and entropy values.

#### `GET /api/v1/emotions/scores/chart?limit=10`
Histogram-style aggregated distribution across emotion classes.

#### `GET /api/v1/emotions/timeline?limit=100`
Chronological emotion records ordered oldest-first, intended for time-series charts.

**Response**:
```json
{
  "user_id": 1,
  "count": 100,
  "timeline": [
    {
      "timestamp": "2026-03-07T14:00:00Z",
      "emotion": "Happiness",
      "confidence": 0.91,
      "emotion_scores": { ... },
      "entropy": 0.42
    }
  ]
}
```

#### `POST /api/v1/emotions/save`
Saves an emotion record. Validates that `emotion_scores` sum to ~1.0 (tolerance ±0.05). Computes and stores Shannon entropy automatically.

**Request body** (`EmotionSaveRequest`):
```json
{
  "dominant_emotion": "Happiness",
  "confidence": 0.91,
  "emotion_scores": { "Happiness": 0.91, "Neutral": 0.06, ... },
  "user_note": "optional string (max 500 chars)",
  "entropy": null
}
```

---

### Analytics — `/api/v1/analytics`

#### `GET /api/v1/analytics/pca`
Computes PCA on all registered and session face embeddings, returns 3D projection for scatter plot visualization.

| Query param | Type | Default |
|-------------|------|---------|
| `include_sessions` | bool | true |
| `session_limit` | int (10–500) | 200 |

**Response**:
```json
{
  "points": [
    {
      "x": -0.42,
      "y": 0.18,
      "z": -0.07,
      "user_id": 1,
      "source": "registered",
      "label": "John Doe",
      "is_current_user": true
    },
    {
      "x": 0.31,
      "y": -0.22,
      "z": 0.14,
      "user_id": 1,
      "source": "session",
      "label": "session_abc123",
      "captured_at": "2026-03-07T14:00:00Z",
      "is_current_user": true
    }
  ],
  "explained_variance": [0.41, 0.22, 0.11],
  "cumulative_variance": [0.41, 0.63, 0.74],
  "total_variance": 0.74,
  "total_points": 45,
  "registered_count": 12,
  "session_count": 33,
  "n_components": 3,
  "embedding_dims": 512
}
```

Requires minimum 3 samples. Returns `{"error": "..."}` with 200 if insufficient data.

#### `POST /api/v1/analytics/session/embed`
Store a 512D embedding captured during a live session for PCA analysis.

**Request body**:
```json
{
  "embedding": [0.01, -0.03, ...],
  "session_id": "optional-session-id"
}
```

Validates L2 norm is 0.99–1.01. Auto-normalizes if outside tolerance.

#### `GET /api/v1/analytics/session/history?limit=50`
Returns captured_at and session_id for recent embeddings (no raw vectors exposed).

#### `DELETE /api/v1/analytics/session`
Deletes all session embeddings for the current user.

---

## WebSocket Stream — `/ws/stream`

**Connection**: `ws://localhost:8000/ws/stream?token=<JWT>`

### Frame Input

The endpoint accepts two frame formats on each receive:

| Format | Description |
|--------|-------------|
| Binary bytes | Raw JPEG — `websocket.send(blob)` from canvas |
| JSON text | `{"image": "data:image/jpeg;base64,..."}` — legacy |

### Processing Pipeline

Per-frame pipeline (each step only runs if prior step succeeded):

1. **Motion detection** — 160×120 grayscale diff, skip if mean < 2.0
2. **Face detection** — SCRFD at threshold 0.3, returns bbox + 5 landmarks
3. **Face geometry** — EAR, MAR, head pose, expression signals from landmarks
4. **Liveness** — MiniFASNetV2 score + EAR composite (both must pass)
5. **Biometric match** — ArcFace embedding → cosine similarity (if enrolled)
6. **Emotion** — EmotiEffLib 8-class probabilities (only if liveness = LIVE)

### Response JSON

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
    "ear": {
      "ear": 0.28,
      "eye_state": "open",
      "is_blinking": false,
      "is_drowsy": false
    },
    "mar": {
      "mar": 0.12,
      "is_yawning": false
    },
    "head_pose": {
      "pitch": -2.1,
      "yaw": 3.4,
      "roll": 0.8,
      "pose_label": "frontal",
      "is_frontal": true,
      "rotation_vector": [0.01, -0.06, 0.02]
    },
    "expressions": {
      "is_smiling": true,
      "is_duchenne_smile": false,
      "is_talking": false,
      "is_happy": true,
      "smile_score": 0.74,
      "talk_score": 0.05,
      "happy_score": 0.82,
      "engagement_score": 0.78
    },
    "attention": {
      "is_distracted": false,
      "attention_state": "focused"
    }
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
    "biometric_match": "MATCH",
    "emotion": "Happiness"
  },

  "analytics": {
    "timestamp": 1741382400.0,
    "ear": 0.28,
    "mar": 0.12,
    "yaw": 3.4,
    "pitch": -2.1,
    "smile_score": 0.74,
    "talk_score": 0.05,
    "happy_score": 0.82,
    "engagement_score": 0.78
  }
}
```

`ml_pipeline.biometric_match` values: `"MATCH"`, `"NO_MATCH"`, `"NOT_AVAILABLE"` (no enrollment), `"NOT_RUN"` (spoof detected).

---

## Services

### `inference_engine.py` — ONNX Singleton

Singleton class loaded at startup. All four models are held as `onnxruntime.InferenceSession` objects.

| Method | Input | Output |
|--------|-------|--------|
| `load_models()` | — | Loads all 4 ONNX sessions |
| `detect_faces(image, threshold=0.5)` | BGR ndarray | `[{bbox, landmarks, score}]` |
| `check_liveness(face_crop)` | BGR crop | float (0–1) |
| `get_face_embedding(aligned_face)` | 112×112 RGB ndarray | float32 ndarray (512,) — L2-normalized |
| `detect_emotion(aligned_face)` | 112×112 RGB ndarray | `{dominant_emotion, confidence, emotion_scores}` |
| `clear_models()` | — | Releases ONNX sessions |

SCRFD input normalization: `(pixel − 127.5) / 128.0`, resized to 640×640.

---

### `face_geometry.py` — Geometric Analysis

All analysis derived from the 5-point SCRFD landmark set (indices: 0=L-eye, 1=R-eye, 2=nose, 3=L-mouth, 4=R-mouth).

**Key constants:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `EAR_BLINK_THRESHOLD` | 0.22 | EAR below this = blinking |
| `EAR_DROWSINESS_THRESHOLD` | 0.18 | EAR below this for 15+ frames = drowsy |
| `DROWSINESS_FRAME_THRESHOLD` | 15 | Consecutive frames for drowsy label |

**Functions:**

| Function | Returns | Notes |
|----------|---------|-------|
| `compute_ear_from_landmarks(landmarks)` | float (0–0.35) | Eye Aspect Ratio via vertical/interocular distances |
| `classify_eye_state(ear, consecutive_frames)` | `{ear, eye_state, is_blinking, is_drowsy}` | eye_state: open / blinking / drowsy / closed |
| `compute_mar_from_landmarks(landmarks)` | float (0–1+) | Mouth Aspect Ratio |
| `estimate_head_pose(landmarks, w, h)` | `{pitch, yaw, roll, pose_label, is_frontal, rotation_vector}` | SolvePnP + Rodrigues → direct Euler extraction |
| `analyze_expression_state(mar, mouth_width, iod, yaw, pitch, ear, prev_mar, mar_series)` | `{expressions, attention}` | Geometric behavioral heuristics |
| `analyze_face_geometry(landmarks, w, h, consec_ear_frames, prev_mar, mar_series)` | Combined dict | High-level entry point for stream pipeline |

Head pose thresholds: yaw ±15°, pitch ±15°, roll ±20°.

**Expression scoring heuristics:**
- `smile_score`: `mouth_width / interocular_distance > 0.55` AND `MAR < 0.40`
- `is_duchenne_smile`: genuine smile + `EAR < 0.30` (eyes narrowing)
- `talk_score`: MAR variance over a `deque(maxlen=6)` temporal window, threshold 0.04
- `engagement_score`: `attention_factor × happy_score` (0 if distracted)

---

### `face_math.py` — Biometric Mathematics

| Function | Signature | Returns |
|----------|-----------|---------|
| `compute_cosine_similarity` | `(vec1, vec2) → float` | Cosine similarity (−1 to 1). Guards against zero vectors. |
| `verify_biometric_match` | `(stored, live, threshold=0.5) → (bool, float)` | Match decision + similarity score |
| `apply_pca_reduction` | `(embeddings: ndarray(N,512), n_components=128) → dict` | Full PCA result dict (see below) |
| `apply_pca_reduction_batch` | `(list[ndarray], n_components) → list[dict]` | Applies PCA to each matrix |

**PCA result dict** (returned by `apply_pca_reduction`):
```python
{
    "reduced_embeddings":      ndarray (N, n_components),
    "principal_components":    ndarray (n_components, 512),
    "explained_variance":      ndarray (n_components,),
    "explained_variance_ratio": ndarray (n_components,),
    "cumulative_variance":     ndarray (n_components,),
    "mean_vector":             ndarray (512,)
}
```

`n_components` is capped at `min(n_samples, n_features)`. If `n_components >= n_features`, returns the original matrix unchanged.

---

### `emotion_math.py` — Affective Mathematics

| Function | Input | Output |
|----------|-------|--------|
| `compute_entropy(probs: list[float])` | 8-element probability vector | Shannon entropy (float) |
| `calculate_russell_coordinates(probs: list[float])` | Emotion probabilities | `{x_coord (valence), y_coord (arousal)}` |

**Russell valence/arousal mappings:**

| Emotion | Valence | Arousal |
|---------|---------|---------|
| Happiness | +0.8 | +0.6 |
| Surprise | +0.3 | +0.8 |
| Neutral | 0.0 | 0.0 |
| Contempt | −0.5 | +0.2 |
| Disgust | −0.7 | +0.2 |
| Fear | −0.6 | +0.7 |
| Anger | −0.7 | +0.8 |
| Sadness | −0.8 | −0.6 |

---

### `analytics.py` — PCA Aggregation Pipeline

Fetches registered and session embeddings from the database, stacks them into a matrix, runs `apply_pca_reduction` with `n_components=3`, and assembles the complete PCA payload for the API response.

Minimum 3 samples required for PCA. Points are labeled with `is_current_user=True` for the requesting user to enable frontend highlighting.

---

## Database Models

### `User`
```python
id, full_name, email, hashed_password
phone_number, gender, age, country
is_active (default True), is_superuser (default False)
face_embedding: Vector(512)          # pgvector, nullable
created_at, updated_at               # auto timestamps
```
Relationships: `emotions` (one-to-many), `session_embeddings` (one-to-many).

### `Emotion`
```python
id, user_id (FK → users, CASCADE)
dominant_emotion (varchar 50, indexed)
confidence (float)
emotion_scores (JSONB, nullable)     # {"Happiness": 0.91, ...}
entropy (float, nullable)
timestamp (auto, indexed)
```

### `FaceSessionEmbedding`
```python
id, user_id (FK → users, CASCADE)
embedding: Vector(512)               # pgvector, indexed
session_id (varchar 64, nullable, indexed)
captured_at (auto, indexed)
```

---

## Alembic Migrations

| Revision | Changes |
|----------|---------|
| `a63ab6c8f35d` | Create `users` table + pgvector extension + `face_embedding Vector(512)` |
| `bb9779230ae7` | Create `emotions` table |
| `8f225089f7e7` | Add `is_superuser` to `users` |
| `ea31fedf1b49` | Create `face_session_embeddings` table |
| `a3c0624aa624` | Add `entropy` column to `emotions` |
| `37fb2ac24236` | Add `country` column to `users` |

Run migrations inside the container:
```bash
docker exec emotrack_backend alembic upgrade head
docker exec emotrack_backend alembic revision --autogenerate -m "description"
```

---

## Testing

Tests run against SQLite in-memory, not PostgreSQL. The test suite is split into unit and integration tests.

```bash
# Run all tests
docker exec emotrack_backend pytest

# Run specific file with output
docker exec emotrack_backend pytest tests/test_face_math.py -v -s

# Run unit tests only
docker exec emotrack_backend pytest tests/test_face_math.py tests/test_face_geometry.py tests/test_inference_engine.py

# Run integration tests only
docker exec emotrack_backend pytest tests/test_integration_auth.py tests/test_integration_users.py tests/test_integration_emotions.py
```

### Critical Test Patterns

**SQLite shared cache** (prevents "no such table" errors):
```python
SQLITE_URL = "sqlite:///:memory:?cache=shared"
engine = create_engine(SQLITE_URL,
    connect_args={"check_same_thread": False, "uri": True},
    poolclass=StaticPool)
```

**Override both `get_db` imports** (routers import from different modules):
```python
from app.core.session import get_db as session_get_db
from app.core.database import get_db as database_get_db

app.dependency_overrides[session_get_db] = override_get_db
app.dependency_overrides[database_get_db] = override_get_db
```

**Empty lifespan** (skip ML loading + PostgreSQL in tests):
```python
@asynccontextmanager
async def _empty_lifespan(_app):
    yield
app.router.lifespan_context = _empty_lifespan
```

### Known Version Pins
`bcrypt==4.0.1` must be pinned — bcrypt ≥ 5.0 is incompatible with passlib 1.7.4 and causes `ValueError: password cannot be longer than 72 bytes`.
