# EMOTITRON

Emotitron is a real-time biometric and emotion intelligence web application.
It captures a user’s live face stream frame, validates that the subject is a real person (anti-spoofing), verifies identity against a registered facial template, detects the current emotion, and stores historical emotion statistics for analytics.

It is designed for authenticated users with a secure profile workflow and biometric-gated sensitive actions.

## What the App Does

1. Registers users with standard account data (name, email, password, etc.).
2. Enrolls a biometric face template (512D embedding) per user.
3. Processes live webcam frames in real time through an ONNX inference pipeline.
4. Detects and returns:
   - Face location
   - Liveness score (real vs spoof)
   - Identity match score
   - Dominant emotion + class probabilities
   - Face geometry metrics (eye state, drowsiness signal, mouth/yawn proxy, head pose)
5. Stores emotion events in PostgreSQL for each authenticated user.
6. Exposes REST analytics endpoints for emotion history, summaries, details, and PCA visualization of face embeddings.
7. Enforces biometric verification before sensitive profile updates (when enrolled).

## Core Technology Stack

### Backend
- FastAPI
- ONNX Runtime
- OpenCV
- SQLAlchemy
- PostgreSQL + pgvector
- JWT authentication

### Models (ONNX)
- Face Detection: SCRFD
- Liveness (Anti-spoof): MiniFASNetV2
- Face Recognition: ArcFace (512D embedding)
- Emotion Recognition: EmotiEffLib / EfficientNet-B0

### Frontend
- React + Vite
- Axios
- TailwindCSS
- Recharts

## How It Works (End-to-End)

### 1. Authentication & Session
- User registers and logs in.
- Backend returns JWT token.
- Frontend sends this token on protected API calls and WebSocket connection.

### 2. Biometric Enrollment
- User uploads/captures a face image.
- Backend:
  - Detects exactly one face.
  - Runs liveness check.
  - Aligns face using landmarks.
  - Extracts normalized 512D ArcFace embedding.
  - Stores embedding in `users.face_embedding` (pgvector).

### 3. Live Stream Inference
- Frontend opens webcam and sends frames over WebSocket.
- For each frame, backend pipeline runs:
  - Face detection
  - Liveness
  - Recognition embedding + cosine similarity against stored template
  - Emotion classification
  - Geometry analysis
- Backend responds with JSON results for UI rendering and telemetry.
- Emotion records are stored in DB for historical analytics.

### 4. Emotion Analytics
- User can query:
  - Paginated history
  - Summary (dominant emotion, counts, percentages, average confidence)
  - Per-emotion detail
  - Emotion scores and chart payloads
  - PCA embedding projection for visualization

### 5. Security Feature
- If user has enrolled biometrics, sensitive profile updates require live biometric verification before saving.

## Data Stored Per User

- Identity data (name, email, password hash, optional profile fields)
- Biometric template (512D vector)
- Emotion events:
  - Dominant emotion
  - Confidence
  - Full class score distribution (JSONB)
  - Timestamp
- Session embeddings for analytics/PCA (optional stream history)

## User Instructions

### A. Create Account
1. Open app.
2. Register with name, email, password.
3. Log in.

### B. Enroll Biometrics (Required for identity matching)
1. Go to Profile.
2. Open camera or upload image.
3. Capture clear frontal face in good lighting.
4. Wait for enrollment success message.

### C. Use Live Inference
1. Open Inference page.
2. Allow camera permissions.
3. Keep face in frame.
4. Watch live outputs:
   - connection/liveness status
   - dominant emotion
   - radar distribution
   - geometry indicators

### D. View Emotion Insights
1. Open Emotions/History pages.
2. Filter and review emotion timeline and confidence.
3. Use Analytics page for aggregate behavior and PCA data.

### E. Update Profile Securely
1. Edit profile fields.
2. If biometrics are enrolled, complete biometric verification modal.
3. Save updates after verification passes.

## Recommended Usage Conditions

- Good front lighting
- Frontal pose
- Single face in frame
- Stable camera (low motion blur)
- Avoid reflective spoof media (screens/photos), as liveness check will reject

## Operational Notes

- No DB migration is needed for frontend/verification flow fixes unless schema changes are introduced.
- Logging is configured to avoid CI import-time filesystem permission issues.
- The app is Dockerized and supports local dev with `docker-compose` or split frontend/backend runs.
