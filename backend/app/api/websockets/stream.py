from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
import json
import numpy as np

from app.core.database import get_db
from app.api.websockets.manager import manager
from app.api.dependencies import get_user_from_token
from app.services.inference_engine import inference_engine
from app.utils.image_processing import decode_base64_image, align_face
from app.services.face_math import verify_biometric_match

router = APIRouter()

@router.websocket("/ws/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    # 1. Autenticación
    user = await get_user_from_token(token, db)
    if not user or not user.is_active:
        await websocket.close(code=1008)
        return

    # 2. Registro en el Manager
    await manager.connect(user.id, websocket)

    try:
        while True:
            # Recibir frame del frontend
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            base64_string = payload.get("image")
            
            if not base64_string:
                continue

            # Decodificar imagen
            image = decode_base64_image(base64_string)
            if image is None:
                continue

            # 3. Pipeline de Machine Learning
            faces = inference_engine.detect_faces(image)
            
            if not faces:
                await manager.send_personal_json({"status": "no_face_detected"}, user.id)
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

            # 4. Liveness Detection
            liveness_score = inference_engine.check_liveness(face_crop)
            is_live = liveness_score > 0.85
            
            response_data = {
                "status": "success",
                "bbox": bbox,
                "liveness": {
                    "is_live": is_live,
                    "score": float(liveness_score)
                }
            }

  
            if is_live:
                
                aligned_face = align_face(image, landmarks)
                
               
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

            
            await manager.send_personal_json(response_data, user.id)

    except WebSocketDisconnect:
        
        manager.disconnect(user.id)
        
    except Exception as e:
        
        print(f"Error en el stream del usuario {user.id}: {str(e)}")
        
        manager.disconnect(user.id)