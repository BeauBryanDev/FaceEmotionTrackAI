from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
import json
import numpy as np
import cv2
#import asynci0
import time 
from starlette.concurrency import run_in_threadpool

from app.core.session import get_db
from app.api.websockets.manager import manager
from app.api.dependencies import get_user_from_token
from app.services.inference_engine import inference_engine
from app.utils.image_processing import decode_base64_image, decode_jpeg_bytes, align_face
from app.services.face_math import verify_biometric_match
#from app.models.emotions import Emotion 
from app.services.face_geometry import analyze_face_geometry
from app.core.logging import get_logger


logger = get_logger(__name__)

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
    previous_gray = None
    MOTION_THRESHOLD = 2.0

    try:
        
        
        while True:
            
            # Receive frame from frontend.
            # Supports both binary JPEG frames and legacy JSON/base64 payloads.
            ws_message = await websocket.receive()
            raw_bytes = ws_message.get("bytes")

            if raw_bytes is not None:
                
                image = decode_jpeg_bytes(raw_bytes)
                
            else:
                
                raw_text = ws_message.get("text")
                
                if not raw_text:
                                        
                    continue
                
                
                payload = json.loads(raw_text)
                
                base64_string = payload.get("image")
                
                if not base64_string:
                    
                    continue
                
                image = decode_base64_image(base64_string)
            
            if image is None:
                
                continue
            
            metrics = {}
            
            ml_pipeline = {
                "face_detected": False,
                "liveness": "NOT_RUN",
                "biometric_match": "NOT_RUN",
                "emotion": "NOT_RUN"
            }

            # Cheap motion detection
            gray_small = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray_small, (160, 120))

            if previous_gray is not None:

                frame_diff = cv2.absdiff(previous_gray, gray_small)
                motion_score = np.mean(frame_diff)

                if motion_score < MOTION_THRESHOLD:
                    # IMPORTANT: always answer to release frontend "waiting" state.
                    # If we just continue here, the client can deadlock after one frame.
                    await websocket.send_json({
                        "status": "skipped_low_motion",
                        "metrics": {"motion_score": float(motion_score)}
                    })
                    continue

            previous_gray = gray_small


            # Decode images 
            start = time.perf_counter()
            # ML Pipeline
            #faces = inference_engine.detect_faces(image , threshold=0.3 ) 
            faces = await run_in_threadpool(
                inference_engine.detect_faces,
                image,
                0.3
            )
                        
            if not faces:
                # Always respond on the same socket that sent this frame.
                await websocket.send_json({"status": "no_face_detected"})
                continue
            
            ml_pipeline["face_detected"] = True
            metrics["face_detection_ms"] = round((time.perf_counter() - start) * 1000, 2)
            
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
            
            start = time.perf_counter()
            # Liveness check - MiniFASNetV2
            liveness_score = await run_in_threadpool(inference_engine.check_liveness, face_crop)

            # COMBINED ANTI-SPOOFING DECISION
            # MiniFASNetV2 alone is insufficient for printed photo attacks on
            # low-resolution laptop cameras (model trained primarily on mobile data).
            # Second layer: geometric liveness via EAR.
            #   - Real faces:     EAR > 0.12 (eyes have depth and structure)
            #   - Printed photos: EAR ~ 0.00 (flat surface, no ocular geometry)
            # add texture variance to liveness score to catch low-res prints/screens
            # Note: bool() cast required - numpy.bool_ is not JSON serializable.
            #is_live = bool(liveness_score > 0.65 and ear_value > 0.125 and texture_variance > 70 )
            metrics["liveness_ms"] = round((time.perf_counter() - start) * 1000, 2)
            #print(f"Liveness | Model: {liveness_score:.4f} | EAR: {ear_value:.4f} | Final: {is_live}")

            is_live = bool( liveness_score > 0.65 and ear_value > 0.125 )  
            logger.debug(
                "Liveness check | model=%.4f texture=%.2f ear=%.4f final=%s",
                liveness_score,
                texture_variance,
                ear_value,
                is_live,
            )
            ml_pipeline["liveness"] = "LIVE" if is_live else "SPOOF"
            
            response_data = {
                
                "status": "success",
                "bbox": bbox,
                "liveness": {
                    "is_live": is_live,
                    "score": float(liveness_score),
                    "texture_score": float(texture_variance)
                }
            }
            logger.info(
                f"Liveness | score={liveness_score:.4f} ear={ear_value:.4f} live={is_live}"
            )

            if is_live:
                
                #aligned_face = align_face(image, landmarks)
                aligned_face_bgr = align_face(image, landmarks)
                aligned_face = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
                
               
                if user.face_embedding is not None:
                    stored_vector = np.array(user.face_embedding, dtype=np.float32)
                    
                    start = time.perf_counter()
                    live_vector = await run_in_threadpool(
                        inference_engine.get_face_embedding,
                        aligned_face
                        )
                    
                    metrics["embedding_ms"] = round((time.perf_counter() - start) * 1000, 2)
                    
                    is_match, similarity = verify_biometric_match(stored_vector, live_vector)
                    
                    response_data["biometrics"] = {
                        "is_match": is_match,
                        "similarity_score": float(similarity)
                    }
                    ml_pipeline["biometric_match"] = "MATCH" if is_match else "NO_MATCH"
                else:
                    
                    response_data["biometrics"] = {"message": "No biometric template found"}
                    ml_pipeline["biometric_match"] = "NOT_AVAILABLE"
                
                start = time.perf_counter()
                
                emotion_result = await run_in_threadpool( inference_engine.detect_emotion,
                                                         aligned_face )
                
                metrics["emotion_ms"] = round((time.perf_counter() - start) * 1000, 2)
                
                ml_pipeline["emotion"] = emotion_result["dominant_emotion"]
                
                response_data["emotion"] = emotion_result
                
            
            response_data["geometry"] = geometry_data
            response_data["metrics"] = metrics
            response_data["ml_pipeline"] = ml_pipeline

            
            # Keep request/response on the same websocket connection to avoid
            # user-id map races during reconnects/dev StrictMode remounts.
            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        
        manager.disconnect(user.id)
        
    except Exception as e:
        logger.exception("Error in user %s stream: %s", user.id, str(e))
        
        manager.disconnect(user.id)


