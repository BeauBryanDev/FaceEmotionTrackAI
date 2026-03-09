# FaceEmotionTrackAI

AI-powered face emotion and biometric tracking system. The backend is a FastAPI service that runs a real-time ML pipeline with ONNX models (face detection, liveness, recognition, emotion), stores 512D embeddings in PostgreSQL + pgvector, and exposes REST + WebSocket APIs. The frontend is React + Vite + Tailwind.

This README reflects a full code review of `backend/` including `app/`, Alembic migrations, tests, and `ml_weights/`.

<p align="center">
  <img src="./frontend/src/assets/emotitrackincon.jpg" alt="Web App Main Icon" width="200">
</p>


## Highlights
- **ML pipeline**: SCRFD (face detection + 5 landmarks), MiniFASNetV2 (liveness), ArcFace (512D embeddings), EmotiEffLib EfficientNet-B0 (emotion).
- **Real-time streaming** via WebSocket `/ws/stream` with per-frame analysis.
- **PostgreSQL + pgvector** for embedding storage and similarity/analytics.
- **Analytics**: PCA reduction of embeddings for 3D scatter visualization.
- **Tests**: 117 test functions under `backend/tests/` (user reports 116 passing).

## Architecture
**Backend stack**
- FastAPI + Uvicorn
- SQLAlchemy 2.x
- PostgreSQL + pgvector
- ONNX Runtime + OpenCV
- JWT auth (OAuth2 password flow)

**Frontend stack**
- React + Vite + Tailwind + Axios

**Services**
- `inference_engine.py`: model loading + inference
- `face_math.py`: cosine similarity + PCA utils
- `face_geometry.py`: EAR/MAR/head-pose per frame
- `analytics.py`: PCA aggregation pipeline

## ML Models
Located in `backend/ml_weights/`:
- `detection/det_500m.onnx` (SCRFD face detection + 5 landmarks)
- `liveness/minifasnet_v2.onnx` (anti-spoofing)
- `recognition/w600k_mbf.onnx` (ArcFace 512D)
- `emotion/emotieff_b0.onnx` (EmotiEffLib EfficientNet-B0)

The models are loaded at startup in FastAPI lifespan (`app/main.py`).

## Database Schema (Alembic)
Tables created by migrations in `backend/alembic/versions/`:
- `users`
  - `id`, `full_name`, `email`, `hashed_password`, `phone_number`, `gender`, `age`
  - `is_active`, `is_superuser`
  - `face_embedding` (pgvector Vector(512))
  - `created_at`, `updated_at`
- `emotions`
  - `id`, `user_id` (FK users), `dominant_emotion`, `confidence`, `emotion_scores` (JSONB), `timestamp`
- `face_session_embeddings`
  - `id`, `user_id` (FK users), `embedding` (Vector(512)), `session_id`, `captured_at`

`init_db()` ensures `CREATE EXTENSION IF NOT EXISTS vector;` at startup.

## Environment Variables
The backend uses `pydantic-settings` and expects:
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `DB_HOST` (use `db` with docker-compose)
- `DB_PORT` (usually `5432`)
- `SECRET_KEY`
- `ENVIRONMENT` (default `development`)

`DATABASE_URL` is **computed** from the above. If you need to override directly, set `DATABASE_URL_OVERRIDE`.

## Running with Docker
`docker-compose.yml` defines `db`, `backend`, and `frontend`.

```bash
docker compose up --build
```

Defaults:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- DB: `localhost:5432`

## Local Backend Run (without Docker)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Make sure your `.env` contains the DB variables and points to a running PostgreSQL instance with pgvector enabled.

## Migrations
```bash
# Inside backend/ with env vars set
alembic -c alembic.ini upgrade head
```

If using Docker:
```bash
docker compose exec backend alembic -c alembic.ini upgrade head
```

## API Overview
All REST endpoints live under `/api/v1`. Most endpoints require a JWT token (Authorization: `Bearer <token>`).

**System**
- `GET /` returns a basic service banner and docs path.
- `GET /api/v1/health` checks DB connectivity and returns `{ status, database, environment }`.

**Auth**
- `POST /api/v1/auth/register` creates a user account. JSON body: `{ full_name, email, password, age?, phone_number?, gender? }`. Returns the user profile (no password, no embedding).
- `POST /api/v1/auth/login` issues a JWT token. Form body (`application/x-www-form-urlencoded`): `username`, `password`. Returns `{ access_token, token_type }`.

**Users** (all require auth)
- `GET /api/v1/users/me` returns the current user profile.
- `PUT /api/v1/users/me` updates profile fields. JSON body supports `full_name`, `email`, `password`, `phone_number`.
- `DELETE /api/v1/users/me` hard-deletes the user and cascades related data.
- `POST /api/v1/users/me/biometrics` registers the biometric template. `multipart/form-data` with `file` (image). Pipeline: detect face → liveness → align → ArcFace embedding → store Vector(512). Returns `{ message }`.
- `POST /api/v1/users/me/face_embedding` currently a no-op that just re-saves the current embedding.
- `DELETE /api/v1/users/me/face_embedding` removes the stored biometric template.

**Emotions** (all require auth)
- `GET /api/v1/emotions/history` returns paginated emotion history. Query: `page`, `page_size`, `emotion_filter`, `date_from`, `date_to`.
- `GET /api/v1/emotions/summary` returns per-emotion counts and average confidence for the current user.
- `GET /api/v1/emotions/details` returns details for one emotion class. Query: `emotion` (one of 8 classes).
- `GET /api/v1/emotions/scores` returns latest records where `emotion_scores` is populated. Query: `limit`.
- `GET /api/v1/emotions/scores/chart` returns a histogram-style distribution from one record with `emotion_scores`.

**Analytics** (all require auth)
- `GET /api/v1/analytics/pca` returns a 3D PCA payload built from registered and session embeddings. Query: `include_sessions` (bool), `session_limit` (10-500).
- `POST /api/v1/analytics/session/embed` stores one session embedding. JSON body: `{ embedding: [512 floats], session_id? }`. Returns the stored record.
- `GET /api/v1/analytics/session/history` returns recent session embeddings. Query: `limit`.
- `DELETE /api/v1/analytics/session` clears all session embeddings for the current user.

## WebSocket Streaming
**Endpoint**: `/ws/stream` (WebSocket)

**Auth**: pass JWT as query param:
```
ws://localhost:8000/ws/stream?token=<JWT>
```

**Client message** (JSON):
```json
{ "image": "data:image/jpeg;base64,..." }
```

**Server response** (example):
```json
{
  "status": "success",
  "bbox": [x1, y1, x2, y2],
  "liveness": { "is_live": true, "score": 0.92 },
  "biometrics": { "is_match": true, "similarity_score": 0.88 },
  "emotion": {
    "dominant_emotion": "Happiness",
    "confidence": 0.93,
    "emotion_scores": { "Anger": 0.01, "...": 0.93 }
  },
  "geometry": {
    "ear": { "ear": 0.28, "eye_state": "open", "is_blinking": false, "is_drowsy": false },
    "mar": { "mar": 0.42, "is_yawning": false },
    "head_pose": { "pitch": 1.2, "yaw": -3.4, "roll": 0.5, "pose_label": "frontal", "is_frontal": true }
  }
}
```

If no face is detected: `{ "status": "no_face_detected" }`.

## Face Geometry (EAR/MAR/Head Pose)
See `backend/app/services/face_geometry.py` and `backend/app/services/FACE_GEOMETRY.md`.

- **EAR (Eye Aspect Ratio)**
  - Blink if EAR < 0.22
  - Drowsy if EAR < 0.18 sustained for 15+ frames
- **MAR (Mouth Aspect Ratio)**
  - Yawn if MAR > 0.60
- **Head pose**: pitch/yaw/roll with simple thresholds

## Tests
All tests live in `backend/tests/`.

What they cover:
- ML utility math: cosine similarity, PCA, embedding norms
- Face geometry calculations (EAR/MAR/head pose)
- Inference engine behavior via mocked ONNX sessions
- Auth + protected routes integration tests
- Emotion history/summary/scores integration tests
- Health endpoint behavior

Counting tests:
- There are **117 `def test_*` functions** in `backend/tests/`.
- I Reports **116 tests passing**.

Run tests locally:
```bash
cd backend
pytest -q
```

## Notable Behaviors (from code)
- Liveness threshold: `0.65` (biometric enrollment + streaming)
- Face detection threshold: `0.5`
- Emotion classes: `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`
- Embeddings are L2-normalized before storage; session embeddings are re-normalized if needed.
- Logging writes to `/app/logs/app.log` in Docker; stdout in dev/CI.

## Project Layout (Backend)
```
backend/
  app/
    api/
      routers/       # REST endpoints
      websockets/    # WebSocket stream
    core/            # config, db, security, logging
    models/          # SQLAlchemy models
    schemas/         # Pydantic schemas
    services/        # ML inference, analytics, geometry
    utils/           # image processing helpers
  alembic/           # migrations
  ml_weights/        # ONNX models
  tests/             # pytest suite
```

## Sample Assets
`backend/faceEmotions/` includes sample face images used for local/manual testing.

## Next Steps (Optional)
1. Generate a `.env` .ENV.EG template and include explicit DB_HOST/DB_PORT defaults for Docker.
2. Add an endpoint for biometric enrollment status (`has_embedding`).
3. Add a scheduled cleanup policy for `face_session_embeddings` if needed.

- Still Working on this Project 'EmotiTrack'
