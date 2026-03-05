from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
import cv2
import numpy as np

from app.api.dependencies import get_current_active_user
from app.core.logging import get_logger
from app.core.session import get_db
from app.models.emotions import Emotion
from app.models.users import User
from app.services.face_math import verify_biometric_match
from app.services.inference_engine import inference_engine
from app.utils.image_processing import align_face


router = APIRouter()
logger = get_logger(__name__)


@router.post("/frame", status_code=status.HTTP_200_OK)
async def infer_frame_emotion(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Runs a single-frame inference pipeline with the 4 ONNX models:
    1) Detection, 2) Liveness, 3) Recognition, 4) Emotion.

    Input:
        multipart/form-data with one image file in `file`.

    Returns:
        JSON payload with bbox, liveness, biometric match info, and current emotion.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not an image.",
        )

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image.",
        )

    try:
        faces = inference_engine.detect_faces(image)
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection model failed.",
        )

    if not faces:
        return {"status": "no_face_detected"}

    primary_face = faces[0]
    bbox = primary_face.get("bbox")
    landmarks = primary_face.get("landmarks")

    img_height, img_width = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid face crop generated from detected bbox.",
        )

    # 1) Liveness
    liveness_score = inference_engine.check_liveness(face_crop)
    is_live = liveness_score > 0.65

    # 2) Recognition
    aligned_face_bgr = align_face(image, landmarks)
    aligned_face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
    live_vector = inference_engine.get_face_embedding(aligned_face_rgb)

    biometrics = {"message": "No biometric template found"}
    if current_user.face_embedding is not None:
        stored_vector = np.array(current_user.face_embedding, dtype=np.float32)
        is_match, similarity = verify_biometric_match(stored_vector, live_vector)
        biometrics = {
            "is_match": bool(is_match),
            "similarity_score": float(similarity),
        }

    # 3) Emotion
    emotion_result = inference_engine.detect_emotion(aligned_face_rgb)

    return {
        "status": "success",
        "bbox": bbox,
        "liveness": {
            "is_live": bool(is_live),
            "score": float(liveness_score),
        },
        "biometrics": biometrics,
        "emotion": emotion_result,
    }

