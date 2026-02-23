from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from sqlalchemy.orm import relationship

class User(Base):
    """
    SQLAlchemy model for the User table.
    Includes standard profile fields and a vector column for biometric data.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    phone_number = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Biometric Master Embedding
    # ArcFace outputs a 512-dimensional normalized vector.
    # The Vector(512) type allows for efficient similarity searches using pgvector.
    face_embedding = Column(Vector(512), nullable=True)
    #One-To-Many relationship
    emotions =  relationship("Emotion", back_populates="user")
    # Metadata for auditing and tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<User(email={self.email}, name={self.full_name})>"
