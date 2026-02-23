from fastapi import APIRouter, Depends, HTTPException, security, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import Any
import numpy as np
import cv2

from app.core.database import get_db
from app.api.dependencies import get_current_active_user
from app.models.users import User
from app.schemas.user_schema import UserResponse , UserUpdate , UserCreate
from app.services.inference_engine import inference_engine
from app.utils.image_processing import align_face, prepare_tensor_for_onnx

router = APIRouter()

@router.get("/me", response_model=UserResponse)
def read_current_user(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Retrieve the profile of the currently authenticated user.
    
    This endpoint requires a valid JWT Bearer token in the Authorization header.
    The get_current_active_user dependency automatically decodes the token,
    queries the database, and injects the User object here.
    
    Returns:
        User: The SQLAlchemy User object, which FastAPI automatically serializes
              into the UserResponse Pydantic schema (excluding sensitive data).
    """
    return current_user


@router.post("/me", status_code=status.HTTP_200_OK)
def create_current_user(
    user_in: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    """
    Create a new user.
    
    This endpoint requires a valid JWT Bearer token in the Authorization header.
    The get_current_active_user dependency automatically decodes the token,
    queries the database, and injects the User object here.
    
    Returns:
        User: The SQLAlchemy User object, which FastAPI automatically serializes
              into the UserResponse Pydantic schema (excluding sensitive data).
    """
    # Create the user record in the database
    new_user = User(
        full_name=user_in.full_name,
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        age=user_in.age
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user 


@router.put("/me", status_code=status.HTTP_200_OK)
def update_current_user(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update the profile of the currently authenticated user.
    
    This endpoint requires a valid JWT Bearer token in the Authorization header.
    The get_current_active_user dependency automatically decodes the token,
    queries the database, and injects the User object here.
    
    Returns:
        User: The SQLAlchemy User object, which FastAPI automatically serializes
              into the UserResponse Pydantic schema (excluding sensitive data).
    """
    
    
    # Update the user record in the database
    for field, value in user_in.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    # Save changes to the database
    
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    
    return current_user 

@router.post("/me/face_embedding", status_code=status.HTTP_200_OK)
def update_face_embedding(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update the face embedding of the currently authenticated user.
    
    This endpoint requires a valid JWT Bearer token in the Authorization header.
    The get_current_active_user dependency automatically decodes the token,
    queries the database, and injects the User object here.
    
    Returns:
        User: The SQLAlchemy User object, which FastAPI automatically serializes
              into the UserResponse Pydantic schema (excluding sensitive data).
    """
    
    # Update the user record in the database
    current_user.face_embedding = current_user.face_embedding
    
    # Save changes to the database
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    
    return current_user


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_user(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> None:
    """
    Soft delete or permanently remove the currently authenticated user.
    
    For compliance with data privacy regulations (like GDPR), users must have 
    the ability to delete their accounts and associated biometric data.
    """
    try:
        # In a production environment, you might prefer a soft delete:
        # current_user.is_active = False
        # db.add(current_user)
        
        # For this MVP, we perform a hard delete to ensure the 512D vector is wiped
        db.delete(current_user)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while attempting to delete the account."
        )
        
    return None


@router.post("/me/biometrics", status_code=status.HTTP_200_OK)
async def register_biometrics(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> dict:
    """
    Registers the master biometric template for the authenticated user.

    This process includes:
    1. Face detection (SCRFD)
    2. Liveness check (Anti-spoofing)
    3. Face alignment (Affine Transformation)
    4. Feature extraction (ArcFace 512D)
    5. Database storage in pgvector
    
    Args:
        file (UploadFile): The uploaded image file (JPEG/PNG).
        current_user (User): The authenticated user session.
        db (Session): The database session.
        
    Returns:
        dict: A success message indicating the biometrics were saved.
    """
    # 1. Validate file format
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not an image."
        )

    # 2. Read and decode the image byte stream into an OpenCV matrix
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode the image. The file may be corrupted."
        )

    # 3. Detect faces using SCRFD
    faces = inference_engine.detect_faces(image)
    
    # 4. Strict Enrollment Validation: Exactly ONE face must be present
    if len(faces) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the provided image."
        )
    if len(faces) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multiple faces detected. Biometric registration requires exactly one face."
        )

    primary_face = faces[0]
    bbox = primary_face.get("bbox")
    landmarks = primary_face.get("landmarks")

    # 5. Crop the face safely for Liveness Detection
    img_height, img_width = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)
    
    face_crop = image[y1:y2, x1:x2]

    # 6. Anti-Spoofing Check
    liveness_score = inference_engine.check_liveness(face_crop)
    if liveness_score < 0.85:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Liveness check failed (Score: {liveness_score:.2f}). Please provide a live capture."
        )

    # 7. Align and Extract the Embedding
    aligned_face = align_face(image, landmarks)
    embedding = inference_engine.get_face_embedding(aligned_face)

    try:
        # 8. Store the 512D vector in the PostgreSQL database
        # SQLAlchemy and pgvector accept standard Python lists for the Vector type
        current_user.face_embedding = embedding.tolist()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save biometric data to the database."
        )

    return {"message": "Biometric master template successfully registered."}