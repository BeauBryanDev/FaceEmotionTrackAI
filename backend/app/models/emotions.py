from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base

class Emotion(Base):
    """
    SQLAlchemy model for storing historical emotion tracking data.
    Represents the mathematical output of the emotion classification network.
    """
    __tablename__ = "emotions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key linking to the users table. 
    # CASCADE ensures that if a user is deleted, their emotion history is also purged.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    dominant_emotion = Column(String(50), nullable=False, index=True)
    
    # The maximum probability score from the Softmax vector max(p)
    confidence = Column(Float, nullable=False)
    
    # Stores the entire probability distribution vector \vec{p} = [p_1, p_2, ..., p_n]
    # JSONB is highly optimized in PostgreSQL for fast indexing and querying
    emotion_scores = Column(JSONB, nullable=True)
    
    # Automatically records the exact time of inference
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationship linking back to the User model
    user = relationship("User", back_populates="emotions")