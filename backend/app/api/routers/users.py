from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any

from app.core.database import get_db
from app.api.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.user import UserResponse

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

@router.put("/me", status_code=status.HTTP_200_OK)
def update_current_user(
    current_user: User = Depends(get_current_active_user),
    user_in: UserResponse,
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


