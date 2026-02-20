from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """
    Shared properties for all User schemas.
    """
    full_name: str = Field(..., min_length=2, max_length=100, description="User's full name")
    email: EmailStr = Field(..., description="Valid email address for authentication")
    age: Optional[int] = Field(None, ge=13, le=120, description="User's age in years")
    phone_number: Optional[str] = Field(None, regex=r'^\+?[1-9]\d{1,14}$', description="Phone number in E.164 format")
    gender: Optional[str] = Field(None, regex=r'^[MF]$', description="User's gender")

class UserCreate(UserBase):
    """
    Schema for user registration.
    Includes the plain text password which will be hashed by the security service.
    """
    password: str = Field(..., min_length=8, description="Strong password, at least 8 characters")

class UserResponse(UserBase):
    """
    Schema for returning user data to the client.
    Strictly excludes the hashed_password and the 512D face_embedding to save bandwidth
    and maintain security.
    """
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        """
        Tells Pydantic to read the data even if it is not a dict, 
        but an ORM model (SQLAlchemy).
        """
        from_attributes = True