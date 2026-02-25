from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from app.core.database import get_db, init_db, SessionLocal
from app.core.session import get_session
from app.services.inference_engine import inference_engine
from app.api.routers import auth, users
from app.api.routers import emotions
from app.api.websockets import stream
from app.core.logging import setup_logging, get_logger



logger = get_logger(__name__)

# --- Application Lifecycle (Lifespan) ---
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Loads Machine Learning ONNX models into memory on startup 
    and clears resources gracefully on shutdown.
    """
    setup_logging()
    print("Initializing database extensions...")
    logger.info("Inicializando extension pgvector en PostgreSQL...")
    init_db()  # Ensure pgvector is set up before any model interactions
    print("Loading ML models into memory...")
    logger.info("Cargando modelos ONNX en memoria...")
    inference_engine.load_models() 
    yield
    print("Cleaning up ML model resources...")
    inference_engine.clear_models()
    logger.info("Aplicacion lista. Docs en http://localhost:8000/docs")
    logger.info("Saliendo...")
# --- FastAPI Instance ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered Face Emotion and Biometric Tracking System",
    version=settings.VERSION,
    lifespan=lifespan
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REST API Routers ---
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(emotions.router, prefix="/api/v1/emotions", tags=["Emotions"])
app.include_router(stream.router, tags=["WebSockets"])

# --- WebSocket Documentation Trick ---
@app.get("/ws/stream", tags=["WebSockets"], summary="WebSocket Stream Documentation")
async def websocket_docs():
    """
    **Note: This endpoint is strictly for documentation purposes.** Swagger UI (OpenAPI 3.0) does not natively render WebSocket connections.
    
    To connect to the real-time biometric stream, establish a WebSocket connection using:
    `ws://localhost:8000/ws/stream?token=YOUR_JWT_TOKEN`
    
    **Payload format (Client -> Server):**
    ```json
    {
        "image": "base64_encoded_string_without_data_uri_header"
    }
    ```
    
    **Response format (Server -> Client):**
    Returns a JSON object containing bounding boxes, liveness scores, 
    biometric similarity, and detected emotions.
    """
    return {
        "message": "Please use a WebSocket client (e.g., frontend code or Postman) to connect via ws:// protocol."
    }

# --- Root Endpoint ---
@app.get("/", tags=["System"])
async def root():
    """Basic root endpoint for service discovery."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "status": "active",
        "docs": "/docs"
    }

# --- Health Check Endpoint ---
@app.get("/api/v1/health", tags=["System"])
async def health_check(db: Session = Depends(get_db)):
    """
    Verifies the operational status of the API and its database connection.
    Confirms that the Docker network (db:5432) is accessible.
    """
    try:
        # Executes a minimal query to verify the connection to the PostgreSQL container
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "environment": settings.ENVIRONMENT
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )