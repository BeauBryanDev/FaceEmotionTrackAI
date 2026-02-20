from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class EmotionBase(BaseModel):
    """
    Shared properties for emotion logs.
    """
    emotion_label: str = Field(
        ..., 
        description="The predicted emotion class (e.g., Happiness, Sadness, Neutral)"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence probability from the HSEmotion EfficientNet model"
    )
    user_note: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Optional contextual note provided by the user"
    )

class EmotionCreate(EmotionBase):
    """
    Schema used when saving a new emotion log after inference.
    """
    pass

class EmotionResponse(EmotionBase):
    """
    Schema for retrieving historical emotion logs for the dashboard.
    """
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True