from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import numpy as np
import time

from app.core.database import get_db
from app.core.logging import get_logger
from app.api.dependencies import get_current_active_user
from app.models.users import User
from app.models.face_session import FaceSessionEmbedding
from app.services.analytics import build_pca_payload

router = APIRouter()
logger = get_logger(__name__)

# REST endpoints for biometric analytics and PCA visualization.
#
# Endpoints:
#   GET  /api/v1/analytics/pca            -> 3D PCA scatter plot payload
#   POST /api/v1/analytics/session/embed  -> store one session embedding
#   GET  /api/v1/analytics/session/history -> session embedding history
#   DELETE /api/v1/analytics/session      -> clear session embeddings for user

# SCHEMAS

class SessionEmbeddingIn(BaseModel):
    """
    Input schema for storing a session embedding captured during a stream.
    The embedding is sent as a flat list of 512 floats from the frontend
    after being received via WebSocket.
    """
    embedding: list[float] = Field(
        ...,
        min_length=512,
        max_length=512,
        description="L2-normalized ArcFace embedding vector of 512 dimensions."
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional session identifier to group embeddings."
    )


class SessionEmbeddingOut(BaseModel):
    """Output schema for a stored session embedding record."""
    id         : int
    user_id    : int
    session_id : Optional[str]
    captured_at: datetime

    class Config:
        from_attributes = True
        
        
# GET /api/v1/analytics/pca
@router.get("/pca", status_code=status.HTTP_200_OK)
async def get_pca_visualization(
    include_sessions: bool = Query(
        default=True,
        description="Include session embeddings alongside registered user embeddings."
    ),
    session_limit: int = Query(
        default=200,
        ge=10,
        le=500,
        description="Maximum number of session embeddings to include in the PCA."
    ),
    current_user: User = Depends(get_current_active_user),
    db: Session        = Depends(get_db)
) -> dict:
    
    """
        Computes PCA on all available face embeddings and returns a 3D scatter
        plot payload for the React analytics dashboard.

        The PCA reduces 512-dimensional ArcFace embeddings to 3 principal
        components using SVD
        
        Points are annotated with user metadata so the frontend can color-code
        them by user identity and highlight the current authenticated user.

        The explained_variance field tells the frontend how much geometric
        information is preserved in the 3D projection.
        
    """
    
    logger.info(
        f"PCA visualization requested.",
        extra={"user_id": current_user.id}
    )
    
    try:
        payload = build_pca_payload(
            db                        = db,
            requesting_user_id        = current_user.id,
            include_session_embeddings= include_sessions,
            session_limit             = session_limit
        )
    except Exception as e:
        logger.error(
            f"PCA computation failed: {e}",
            extra={"user_id": current_user.id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PCA computation failed. See server logs for details."
        )

    if "error" in payload:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=payload["error"]
        )

    logger.info(
        
        f"PCA payload built. points={payload['total_points']} "
        f"variance={payload['total_variance']}",
        extra={"user_id": current_user.id}
    )

    return payload


# POST /api/v1/analytics/session/embed
@router.post(
    "/session/embed",
    response_model=SessionEmbeddingOut,
    status_code=status.HTTP_201_CREATED
)
async def store_session_embedding(
    body    : SessionEmbeddingIn,
    current_user   : User    = Depends(get_current_active_user),
    db   : Session = Depends(get_db)
) -> FaceSessionEmbedding:

    """
    Stores a single 512D ArcFace embedding captured during a live session.

    This endpoint is called by the frontend after receiving an embedding
    from the WebSocket stream. The session_id groups embeddings from the
    same connection for time-series analysis.

    The stored embeddings feed the PCA analytics endpoint to populate
    the 3D latent space scatter plot with real-time data points.
    """
    
    embedding_array = np.array(body.embedding, dtype=np.float32)
    
    # Validate L2 norm - embeddings from ArcFace should be unit vectors.
    norm = float(np.linalg.norm(embedding_array))
    
    if norm < 0.99 or norm > 1.01:
        
        logger.warning(
            
            f"Received embedding with non-unit norm: {norm:.4f}. "
            f"Re-normalizing before storage.",
            extra={"user_id": current_user.id}
        )
        
        if norm > 0:
            
            embedding_array = embedding_array / norm

    # Generate a session_id if not provided by the client.
    session_id = body.session_id or f"{current_user.id}_{int(time.time())}"

    new_record = FaceSessionEmbedding(
        
        user_id   = current_user.id,
        embedding = embedding_array.tolist(),
        session_id= session_id
    )
    
    try:
        
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        
    except Exception as e:
        
        db.rollback()
        
        logger.error(
            
            f"Failed to store session embedding: {e}",
            extra={"user_id": current_user.id},
            exc_info=True
        )
        raise HTTPException(
            
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store session embedding."
        )

    logger.info(
        
        f"Session embedding stored. id={new_record.id} session={session_id}",
        extra={"user_id": current_user.id}
    )

    return new_record


# GET /api/v1/analytics/session/history

@router.get("/session/history", status_code=status.HTTP_200_OK)
async def get_session_embedding_history(
    
    limit        : int     = Query(default=50, ge=1, le=200),
    current_user : User    = Depends(get_current_active_user),
    db           : Session = Depends(get_db)
) -> dict:
    
    """
    Returns the most recent session embeddings for the authenticated user.
    Useful for debugging the PCA data pipeline and auditing captured vectors.
    """
    
    records = (
        
        db.query(FaceSessionEmbedding)
        .filter(FaceSessionEmbedding.user_id == current_user.id)
        .order_by(FaceSessionEmbedding.captured_at.desc())
        .limit(limit)
        .all()
    ) # List of FaceSessionEmbedding records
    
    return {
        
        "user_id": current_user.id,
        "count"  : len(records),
        "records": [
            {
                "id"        : r.id,
                "session_id": r.session_id,
                "captured_at": r.captured_at.isoformat(),
            }
            for r in records
        ]
    }
    

# DELETE /api/v1/analytics/session
@router.delete("/session", status_code=status.HTTP_200_OK)
async def clear_session_embeddings(
    
    current_user : User    = Depends(get_current_active_user),
    db           : Session = Depends(get_db)
) -> dict:
    
    """
    Deletes all session embeddings for the authenticated user.
    Useful for resetting the PCA scatter plot or complying with
    data retention policies (GDPR right to erasure).
    """
    
    try:
        
        deleted_count = (
            
            db.query(FaceSessionEmbedding)
            .filter(FaceSessionEmbedding.user_id == current_user.id)
            .delete()
        )
        db.commit()
        
    except Exception as e:
        
        db.rollback()
        
        logger.error(
            
            f"Failed to clear session embeddings: {e}",
            extra={"user_id": current_user.id},
            exc_info=True
        )
        
        raise HTTPException(
            
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear session embeddings."
        )
        
    logger.info(
        
        f"Session embeddings cleared. deleted={deleted_count}",
        extra={"user_id": current_user.id}
    )
    
    return {
        
        "message": f"Deleted {deleted_count} session embeddings.",
        "user_id": current_user.id,
        "count"  : deleted_count
    }