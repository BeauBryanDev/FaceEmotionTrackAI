from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db

from app.api.routers import auth, users, emotions
from app.api.websockets import stream
from app.core.config import settings
from app.services.inference_engine import inference_engine
from app.api.routers import auth, users, emotions
from app.api.websockets import stream


# --- Ciclo de Vida (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esto ocurre cuando el servidor arranca
    print("Loading models in Memory..")
    inference_engine.load_models() 
    yield
    # Esto ocurre cuando el servidor se apaga
    print("Cleaning resources..")
    inference_engine.clear_models()

# --- Instancia de FastAPI ---
app = FastAPI(
    title="FaceEmotionTrackAI API",
    description="MVP para tracking de emociones y biometría facial",
    version="1.0.0",
    lifespan=lifespan
)

# --- Configuración de CORS ---
# Fundamental para que React (puerto 3000) pueda hablar con FastAPI (puerto 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción pondrás tu dominio de EC2
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Inclusión de Routers (REST) ---
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
#app.include_router(emotions.router, prefix="/api/v1/emotions", tags=["Emotions"])

# --- Endpoint de WebSockets (Real-time Video Inference) ---
app.add_api_websocket_route("/ws/stream", stream.websocket_endpoint)

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {
        "message": "Welcome to FaceEmotionTrackAI API",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/api/v1/health", tags=["System"])
async def health_check(db: Session = Depends(get_db)):
    """
    Verifies the operational status of the API and its database connection.
    Confirms that the Docker network (db:5432) is accessible.
    """
    try:
        # Executes a minimal query to verify the connection to the 'db' container
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