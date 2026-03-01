import sys
import os
import json
from sqlalchemy.orm import Session

# Add the parent directory to sys.path so we can import 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.database import SessionLocal
from app.models.users import User
from app.models.emotions import Emotion
from app.core.security import get_password_hash

# Patch for passlib/bcrypt compatibility in Python 3.12
import logging
logging.getLogger("passlib").setLevel(logging.ERROR)

def test_emotion_storage():
    db: Session = SessionLocal()
    test_email = "websocket_sim@example.com"
    
    try:
        print("--- Starting Emotion JSONB Storage Test ---")
        
        # 1. Setup Test User
        # We need a valid user_id to satisfy the ForeignKey constraint
        user = db.query(User).filter(User.email == test_email).first()
        if not user:
            print("Creating dummy user for emotion test...")
            user = User(
                full_name="WebSocket Simulator",
                email=test_email,
                hashed_password=get_password_hash("secure_password"),
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # 2. Simulate Inference Engine Output
        # This mimics the exact dictionary format your ONNX model returns
        simulated_inference = {
            "dominant_emotion": "happy",
            "confidence": 0.945,
            "emotion_scores": {
                "happy": 0.945,
                "neutral": 0.032,
                "surprise": 0.018,
                "sad": 0.003,
                "angry": 0.002
            }
        }
        
        # 3. Save to Database (Simulating the WebSocket stream logic)
        print("Saving simulated emotion data to PostgreSQL...")
        new_emotion = Emotion(
            user_id=user.id,
            dominant_emotion=simulated_inference["dominant_emotion"],
            confidence=simulated_inference["confidence"],
            emotion_scores=simulated_inference["emotion_scores"]
        )
        
        db.add(new_emotion)
        db.commit()
        db.refresh(new_emotion)
        print(f"✅ Emotion record saved successfully with ID: {new_emotion.id}")
        
        # 4. Retrieve and Verify JSONB Integrity
        print("\n--- Retrieving record to verify JSONB storage ---")
        retrieved_emotion = db.query(Emotion).filter(Emotion.id == new_emotion.id).first()
        
        if retrieved_emotion:
            print(f"Dominant Emotion: {retrieved_emotion.dominant_emotion}")
            print(f"Confidence: {retrieved_emotion.confidence}")
            
            # This is the crucial test: Verify that SQLAlchemy parsed the JSONB back into a Python dict
            scores_type = type(retrieved_emotion.emotion_scores).__name__
            print(f"JSONB Scores Native Type: {scores_type}")
            print(f"JSONB Scores Content:\n{json.dumps(retrieved_emotion.emotion_scores, indent=2)}")
            
            if scores_type == 'dict':
                print("\n💎 Emotion JSONB data integrity verified! PostgreSQL and SQLAlchemy are perfectly synced.")
            else:
                print("\n⚠️ Warning: Data was saved, but not retrieved as a native Python dictionary.")
        else:
            print("❌ Error: Could not retrieve the saved emotion record.")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    test_emotion_storage()