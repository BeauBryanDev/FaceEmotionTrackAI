from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
import json
import numpy as np
import cv2
#import asyncio

from app.core.session import get_db
from app.api.websockets.manager import manager
from app.api.dependencies import get_user_from_token
from app.services.inference_engine import inference_engine
from app.utils.image_processing import decode_base64_image, align_face
from app.services.face_math import verify_biometric_match
#from app.models.emotions import Emotion 
from app.services.face_geometry import analyze_face_geometry


router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    # 1. Authentication
    user = await get_user_from_token(token, db)
    
    if not user or not user.is_active:
        
        await websocket.close(code=1008)
        
        return

    # 2. Register Connection in Manager
    await manager.connect(user.id, websocket)
    
    consecutive_low_ear_frames = 0

    try:
        
        while True:
            # Receivee frame from frontend
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            base64_string = payload.get("image")
            
            if not base64_string:
                continue

            # Decodificar imagen
            image = decode_base64_image(base64_string)
            if image is None:
                continue

            # ML Pipeline
            faces = inference_engine.detect_faces(image , threshold=0.3 ) 
            
            if not faces:
                # Always respond on the same socket that sent this frame.
                await websocket.send_json({"status": "no_face_detected"})
                continue

            
            primary_face = faces[0]
            bbox = primary_face.get("bbox")
            landmarks = primary_face.get("landmarks")
            
            
            img_height, img_width = image.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            img_h, img_w = image.shape[:2]
            
            geometry_data = analyze_face_geometry(
                landmarks=np.array(landmarks),
                image_width=img_w,
                image_height=img_h,
                consecutive_low_ear_frames=consecutive_low_ear_frames
            )
            
            if geometry_data["ear"]["ear"] < 0.22:
                
                
                consecutive_low_ear_frames += 1
                
            else:
                
                consecutive_low_ear_frames = 0
            
            
            # Texture Analysis (Laplacian Variance) - Catch low-res prints/screens
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Extract EAR from geometry data already computed above
            ear_value = geometry_data["ear"]["ear"]

            # Liveness check - MiniFASNetV2
            liveness_score = inference_engine.check_liveness(face_crop)

            # COMBINED ANTI-SPOOFING DECISION
            # MiniFASNetV2 alone is insufficient for printed photo attacks on
            # low-resolution laptop cameras (model trained primarily on mobile data).
            # Second layer: geometric liveness via EAR.
            #   - Real faces:     EAR > 0.12 (eyes have depth and structure)
            #   - Printed photos: EAR ~ 0.00 (flat surface, no ocular geometry)
            # Note: bool() cast required - numpy.bool_ is not JSON serializable.
            #is_live = bool(liveness_score > 0.65 and ear_value > 0.125 and texture_variance > 70 )

            #print(f"Liveness | Model: {liveness_score:.4f} | EAR: {ear_value:.4f} | Final: {is_live}")

            is_live = bool( liveness_score > 0.65 and ear_value > 0.125 )  

            print(f"Liveness Check | Model: {liveness_score:.4f} | Texture: {texture_variance:.2f} | EAR : {ear_value:.4f} | Final: {is_live}")
    
            response_data = {
                "status": "success",
                "bbox": bbox,
                "liveness": {
                    "is_live": is_live,
                    "score": float(liveness_score),
                    "texture_score": float(texture_variance)
                }
            }

            if is_live:
                
                #aligned_face = align_face(image, landmarks)
                aligned_face_bgr = align_face(image, landmarks)
                aligned_face = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
                
               
                if user.face_embedding is not None:
                    stored_vector = np.array(user.face_embedding, dtype=np.float32)
                    live_vector = inference_engine.get_face_embedding(aligned_face)
                    
                    is_match, similarity = verify_biometric_match(stored_vector, live_vector)
                    
                    response_data["biometrics"] = {
                        "is_match": is_match,
                        "similarity_score": float(similarity)
                    }
                else:
                    response_data["biometrics"] = {"message": "No biometric template found"}

                
                emotion_result = inference_engine.detect_emotion(aligned_face)
                response_data["emotion"] = emotion_result
                
            
            response_data["geometry"] = geometry_data
            
            # Keep request/response on the same websocket connection to avoid
            # user-id map races during reconnects/dev StrictMode remounts.
            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        
        manager.disconnect(user.id)
        
    except Exception as e:
        
        print(f"Error en el stream del usuario {user.id}: {str(e)}")
        
        manager.disconnect(user.id)



# before it used to save emotion in the database, now it is only returning the emotion result to the client
#async def save_emotion(emotion_result, user):
#  try:
#                     # Extract data from inference engine's output dictionary
#                     dominant = emotion_result.get("dominant_emotion", "Neutral")
#                     confidence = emotion_result.get("confidence", 0.0)
#                     scores = emotion_result.get("emotion_scores", {}) #TODO , it has to be changed, I need to return all emotions scores, all of them.  ||  None . If None, it will be stored as null in PostgreSQL, if dict, it will be stored as JSONB.

#                     # Create the record using the SQLAlchemy model
#                     new_emotion_record = Emotion(
#                         user_id=user.id,
#                         dominant_emotion=dominant,
#                         confidence=float(confidence),  # Cast to float to avoid numpy type issues
#                         emotion_scores=scores          # PostgreSQL will automatically cast this dict to JSONB
#                     )
#                     # db thread is separate from the main event loop, so we use run_in_executor to avoid blocking
#                     db.add(new_emotion_record)
            
#                     # DO NOT commit in the main thread, it blocks the WebSocket loop.
#                     # We ONLY use the executor for the commit.
#                     await asyncio.get_event_loop().run_in_executor(None, db.commit)
            
#                 except Exception as db_error:
#                     # Rollback the transaction on error so the session isn't poisoned
#                     db.rollback()
#                     print(f"Failed to save emotion to DB for user {user.id}: {str(db_error)}")
