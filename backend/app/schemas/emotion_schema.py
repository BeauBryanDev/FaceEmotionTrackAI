from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class EmotionBase(BaseModel):
    """
    Shared properties for emotion logs.
    """
    dominant_emotion: str = Field(
        ..., 
        description="The predicted emotion class (e.g., Happiness, Sadness, Neutral)"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence probability from the HSEmotion EfficientNet model"
    )
    emotion_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Full probability distribution across all detected emotion classes."
    )
    
    user_note: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Optional contextual note provided by the user"
    )

class EmotionCreate(BaseModel):
    """
    Schema used when saving a new emotion log after inference.
    """
    emotion_label: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    emotion_scores: Optional[Dict[str, float]] = None

class EmotionResponse(EmotionBase):
    """
    Schema for retrieving historical emotion logs for the dashboard.
    """
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True