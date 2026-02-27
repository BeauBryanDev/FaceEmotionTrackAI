# FaceEmotionTrackAI - Project Guidelines for Claude

> **Note**: This project was started and developed by the project owner before Claude's involvement. Claude is assisting with testing, debugging, and improvements.

## Project Overview

**FaceEmotionTrackAI** is an AI-powered Face Emotion and Biometric Tracking System that provides real-time face detection, emotion recognition, liveness detection, and biometric similarity analysis.

### Tech Stack
- **Backend**: FastAPI (Python 3.12)
- **Database**: PostgreSQL with pgvector extension
- **ML Models**: ONNX Runtime (detection, recognition, liveness, emotion)
- **Frontend**: React (assumed, WebSocket connection)
- **Computer Vision**: OpenCV, NumPy
- **Deployment**: Docker containers

### Architecture
- RESTful API with WebSocket support for real-time streaming
- Microservices architecture (backend + PostgreSQL containers)
- ML inference engine with 4 ONNX models loaded in memory
- Vector similarity search for face embeddings (512D ArcFace)
- PCA analytics for dimensionality reduction

## Directory Structure

```
FaceEmotionTrackAI/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routers/          # REST endpoints (auth, users, emotions, analytics)
│   │   │   └── websockets/       # WebSocket stream endpoint
│   │   ├── core/                 # Config, database, security, logging
│   │   ├── models/               # SQLAlchemy ORM models (users, emotions, face_session)
│   │   ├── services/             # Business logic (inference_engine, face_geometry, etc.)
│   │   └── main.py              # FastAPI app entry point
│   ├── tests/
│   │   ├── conftest.py          # Shared pytest fixtures
│   │   ├── test_integration_auth.py
│   │   └── test_*.py            # Unit and integration tests
│   ├── requirements.txt
│   └── Dockerfile
├── privates/                     # Private notes and documentation
└── CLAUDE.md                    # This file
```

## Critical Rules - DO NOT VIOLATE

### 🚫 Absolutely NO modifications to `app/` directory
- **NEVER** edit production code in `backend/app/` unless explicitly requested
- All test fixes must be in `backend/tests/` only
- Use mocking, patching, and dependency overrides instead
- If a bug is found in app/, report it to the user first

### ⚠️ Code Quality Standards
- **No over-engineering**: Only make changes directly requested
- **No premature optimization**: Don't add features "for the future"
- **No unnecessary abstractions**: Three similar lines > premature abstraction
- **No extra error handling**: Only validate at system boundaries
- **No extra comments/docstrings**: Only add to code you create/modify
- Keep solutions simple and focused

### 🔒 Security Guidelines
- Never commit secrets, .env files, or credentials
- Always validate user input at API boundaries
- Use parameterized queries (SQLAlchemy ORM does this)
- No SQL injection, XSS, or command injection vulnerabilities
- Follow OWASP Top 10 best practices

## Testing Guidelines

### Running Tests
```bash
# Run all tests
docker exec emotrack_backend pytest

# Run specific test file
docker exec emotrack_backend pytest tests/test_integration_auth.py

# Run with verbose output
docker exec emotrack_backend pytest tests/test_integration_auth.py -v

# Run with output capture disabled (see print statements)
docker exec emotrack_backend pytest tests/test_integration_auth.py -v -s
```

### Test Database Setup (SQLite)
Integration tests use an in-memory SQLite database:

```python
# IMPORTANT: Use shared in-memory DB with StaticPool
SQLITE_URL = "sqlite:///:memory:?cache=shared"

engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False, "uri": True},
    poolclass=StaticPool,  # Ensures all connections share same DB
)
```

**Why?** Regular `sqlite://` creates a separate database per connection!

### Handling pgvector in Tests
PostgreSQL-specific types must be patched for SQLite:

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

# Replace Vector(512) with JSON text storage for SQLite
for mapper in Base.registry.mappers:
    for column in mapper.mapped_table.columns:
        if isinstance(column.type, Vector):
            column.type = VectorAsText()  # Custom TypeDecorator
        elif isinstance(column.type, JSONB):
            column.type = Text()
```

### Mocking the App Lifespan
TestClient triggers the app's lifespan which loads ML models and connects to PostgreSQL. Override it:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def _empty_lifespan(_app):
    """Empty lifespan for testing - skips DB init and ML model loading."""
    yield

# Apply BEFORE creating TestClient
app.router.lifespan_context = _empty_lifespan
```

## Known Issues & Solutions

### Issue: "no such table: users" in tests
**Cause**: Using `sqlite://` creates separate DBs per connection
**Solution**: Use `sqlite:///:memory:?cache=shared` with `StaticPool`
**Reference**: `privates/test_int_auth_fix.txt`

### Issue: bcrypt ValueError (password > 72 bytes)
**Cause**: Incompatibility between passlib 1.7.4 and bcrypt 5.0.0
**Solution**: Pin `bcrypt==4.0.1` in requirements.txt

### Issue: Tables not visible across connections
**Cause**: SQLite in-memory DB isolation
**Solution**: Use shared cache + StaticPool (see test_integration_auth.py)

## Database Models

### Key Models
1. **User** (`app/models/users.py`)
   - Stores user credentials and face embeddings (512D vector)
   - Fields: id, full_name, email, hashed_password, face_embedding, age, gender

2. **Emotion** (`app/models/emotions.py`)
   - Emotion detection records linked to users
   - Stores emotion probabilities and timestamps

3. **FaceSessionEmbedding** (`app/models/face_session.py`)
   - Stores multiple face embeddings per user for PCA analysis
   - Links to User model via user_id

### Database Migrations
```bash
# Create new migration
docker exec emotrack_backend alembic revision --autogenerate -m "description"

# Run migrations
docker exec emotrack_backend alembic upgrade head
```

## ML Models & Inference

### Models Located: `backend/ml_weights/`
1. **detection.onnx** - Face detection (YuNet or similar)
2. **recognition.onnx** - ArcFace face embeddings (512D)
3. **liveness.onnx** - Anti-spoofing / liveness detection
4. **emotion.onnx** - 7 emotion classification

### Inference Engine
File: `app/services/inference_engine.py`
- Singleton pattern (`inference_engine` instance)
- Loads all models on startup via lifespan
- Provides `detect()`, `recognize()`, `check_liveness()`, `detect_emotion()`

### Face Geometry Analysis
File: `app/services/face_geometry.py`
- Estimates head pose (yaw, pitch, roll) from 5-point landmarks
- Calculates EAR (Eye Aspect Ratio) for blink detection
- Calculates MAR (Mouth Aspect Ratio) for mouth opening
- Uses **direct Euler angle extraction** from rotation matrix (NOT decompose
HomographyMat - that was buggy!)

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - JWT token login

### Users
- `GET /api/v1/users/me` - Get current user (protected)

### Emotions
- `POST /api/v1/emotions/` - Record emotion detection
- `GET /api/v1/emotions/user/{user_id}` - Get user's emotion history

### Analytics
- `GET /api/v1/analytics/*` - PCA and similarity analytics

### WebSocket
- `WS /ws/stream` - Real-time face tracking stream
  - Requires JWT token via query param: `?token=<access_token>`
  - Returns JSON with bounding boxes, liveness, emotions, similarity

## Recent Work Done

### Feb 27, 2026 - Integration Auth Tests Fixed
- Fixed "no such table: users" error in integration tests
- Root cause: SQLite in-memory database connection isolation
- Solution: Switched to `sqlite:///:memory:?cache=shared` with StaticPool
- Overrode app lifespan to prevent PostgreSQL connection in tests
- Pinned bcrypt==4.0.1 for passlib compatibility
- **Result**: All 13 integration auth tests now pass ✅

### Prior Work (by owner)
- Unit tests for face_geometry (estimate_head_pose, EAR, MAR)
- Found 5 real bugs through unit testing
- Replaced decompose
HomographyMat with direct Euler angle extraction
- Added PCA analytics to SCD (Stored Consolidated Data)
- Refactored stream.py for PCA-EAR handling
- Set up conftest.py with fixtures for embeddings, landmarks, images

## Development Workflow

### Making Changes
1. Read existing code before modifying
2. Understand the current architecture
3. Make minimal, focused changes
4. Write/update tests if needed
5. Run tests to verify: `docker exec emotrack_backend pytest`
6. Ask user before committing if uncertain

### Debugging
1. Check container logs: `docker logs emotrack_backend`
2. Exec into container: `docker exec -it emotrack_backend bash`
3. Use pytest with `-v -s` for detailed output
4. Add strategic print statements (remove after debugging)
5. Check database state: `docker exec emotrack_backend psql -U beauAdmin -d emotrack`

### Git Workflow
User has uncommitted changes:
- Modified: `backend/requirements.txt`
- Untracked: `backend/tests/test_integration_auth.py`

**Before committing, ask the user if they want to:**
- Review changes first
- Add/modify commit message
- Stage specific files

## Dependencies & Versions

### Critical Version Pins
```
Python: 3.12
FastAPI: 0.110.0
SQLAlchemy: 2.0.35
pgvector: 0.3.6
bcrypt: 4.0.1  # For passlib 1.7.4 compatibility
passlib[bcrypt]: 1.7.4
pytest: 7.4.3
onnxruntime: 1.20.0
numpy: 1.26.4
opencv-python-headless: 4.10.0.84
```

See `backend/requirements.txt` for full list.

## Contact & Collaboration

- **Owner**: beaunix
- **Project Started**: Before Claude's involvement (2026)
- **Claude's Role**: Testing, debugging, code review, improvements

### When to Ask the User
- Before modifying production code (app/)
- Before committing changes
- When uncertain about requirements
- When finding bugs in production code
- Before adding new dependencies
- Before making architectural changes

## Resources

- **Test Fix Notes**: `privates/test_int_auth_fix.txt`
- **Container Name**: `emotrack_backend`
- **Database Container**: `db` (PostgreSQL)
- **Database URL**: `postgresql://beauAdmin:***@db:5432/emotrack`

---

**Remember**: This project was built with care by its owner. Claude's job is to assist, not to rebuild or over-engineer. Keep changes minimal, focused, and respectful of the existing architecture.
