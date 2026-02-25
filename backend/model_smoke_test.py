import cv2
import numpy as np
import os
import sys

from app.services.inference_engine import inference_engine
from app.utils.image_processing import align_face

# ------------------------------------------------------------------ #
#  ANSI colors - funcionan en terminales Linux/Mac y en la mayoria
#  de terminales Windows modernas.
# ------------------------------------------------------------------ #
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}[OK]{RESET}    {msg}")
def fail(msg: str) -> None: print(f"  {RED}[FAIL]{RESET}  {msg}")
def info(msg: str) -> None: print(f"  {CYAN}[INFO]{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}[WARN]{RESET}  {msg}")

SEPARATOR = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ------------------------------------------------------------------ #
#  1. CARGA DE IMAGEN
# ------------------------------------------------------------------ #
def load_image(image_path: str) -> np.ndarray:
    section("PASO 1 - Carga de imagen")

    if not os.path.exists(image_path):
        fail(f"Imagen no encontrada: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        fail(f"cv2.imread no pudo decodificar: {image_path}")
        sys.exit(1)

    h, w, c = img.shape
    ok(f"Imagen cargada correctamente.")
    info(f"Path  : {image_path}")
    info(f"Shape : H={h} W={w} C={c}  (formato BGR de OpenCV)")
    return img


# ------------------------------------------------------------------ #
#  2. CARGA DE MODELOS ONNX
# ------------------------------------------------------------------ #
def load_models() -> None:
    section("PASO 2 - Carga de modelos ONNX")
    try:
        inference_engine.load_models()
        ok("Todos los modelos ONNX cargados en memoria.")
    except FileNotFoundError as e:
        fail(f"Archivo de modelo no encontrado: {e}")
        fail("Verifica que ml_weights/ tenga los 4 archivos .onnx")
        sys.exit(1)
    except Exception as e:
        fail(f"Error inesperado al cargar modelos: {e}")
        sys.exit(1)


# ------------------------------------------------------------------ #
#  3. INSPECCION DE SHAPES DE LOS 4 MODELOS
# ------------------------------------------------------------------ #
def inspect_model_shapes() -> None:
    section("PASO 3 - Inspeccion de inputs/outputs de cada modelo")

    models = ["detection", "liveness", "recognition", "emotion"]

    for model_name in models:
        print(f"\n  Modelo: {model_name.upper()}")
        session = inference_engine.get_session(model_name)
        if session is None:
            fail(f"Sesion '{model_name}' no disponible.")
            continue

        for inp in session.get_inputs():
            info(f"  INPUT  -> name='{inp.name}'  shape={inp.shape}  type={inp.type}")
        for out in session.get_outputs():
            info(f"  OUTPUT -> name='{out.name}'  shape={out.shape}  type={out.type}")


# ------------------------------------------------------------------ #
#  4. DETECCION DE ROSTROS (SCRFD)
# ------------------------------------------------------------------ #
def test_detection(img: np.ndarray) -> list:
    section("PASO 4 - Face Detection (SCRFD)")

    faces = inference_engine.detect_faces(img)
    total = len(faces)

    if total == 0:
        warn("No se detecto ningun rostro.")
        warn("Prueba con otra imagen de faceEmotions/")
        return []

    ok(f"{total} rostro(s) detectado(s).")

    for i, face in enumerate(faces):
        bbox      = face["bbox"]
        score     = face["score"]
        landmarks = face["landmarks"]
        x1, y1, x2, y2 = [round(v, 1) for v in bbox]
        info(f"  Rostro {i+1}: bbox=[{x1},{y1},{x2},{y2}]  conf={score:.4f}")
        info(f"            landmarks shape={landmarks.shape}  (esperado: (5,2))")

    return faces


# ------------------------------------------------------------------ #
#  5. LIVENESS DETECTION (MiniFASNetV2)
#     FIX aplicado: face_crop se convierte a RGB antes de pasar al modelo
# ------------------------------------------------------------------ #
def test_liveness(img: np.ndarray, bbox: list) -> float:
    section("PASO 5 - Liveness Detection (MiniFASNetV2)")

    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        fail("El crop del rostro esta vacio. BBox fuera de los limites de la imagen.")
        return 0.0

    info(f"Face crop shape para liveness: {face_crop.shape}  (BGR, sera convertido a RGB internamente)")

    liveness_score = inference_engine.check_liveness(face_crop)

    info(f"Liveness score (prob_real): {liveness_score:.4f}")

    # Verificacion del orden de clases MiniFASNetV2
    # index 1 = Real segun el checkpoint de yakhyo/face-anti-spoofing
    if liveness_score > 0.5:
        ok(f"Clasificado como REAL  (score={liveness_score:.4f} > 0.5)")
        ok("Orden de clases confirmado: index 1 = Real")
    else:
        warn(f"Clasificado como FAKE  (score={liveness_score:.4f} <= 0.5)")
        warn("Si la imagen es de una persona real, el orden de clases puede estar invertido.")
        warn("En inference_engine.check_liveness cambia probabilities[1] a probabilities[0]")

    # Evaluacion contra el threshold operacional del stream (0.60)
    STREAM_THRESHOLD     = 0.60
    BIOMETRIC_THRESHOLD  = 0.70

    if liveness_score >= BIOMETRIC_THRESHOLD:
        ok(f"Pasa threshold de registro biometrico ({BIOMETRIC_THRESHOLD})")
    elif liveness_score >= STREAM_THRESHOLD:
        ok(f"Pasa threshold de stream en tiempo real ({STREAM_THRESHOLD})")
        warn(f"No alcanza threshold de registro biometrico ({BIOMETRIC_THRESHOLD})")
    else:
        fail(f"No pasa ninguno de los thresholds operacionales.")
        fail(f"  Stream:     {STREAM_THRESHOLD}  (actual: {liveness_score:.4f})")
        fail(f"  Biometrico: {BIOMETRIC_THRESHOLD}  (actual: {liveness_score:.4f})")

    return liveness_score


# ------------------------------------------------------------------ #
#  6. ALINEACION + FACE RECOGNITION (ArcFace MobileFaceNet)
#     FIX aplicado: aligned_face se convierte de BGR a RGB antes de ArcFace
# ------------------------------------------------------------------ #
def test_recognition(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    section("PASO 6 - Face Alignment + ArcFace Recognition")

    # align_face retorna BGR (opera sobre la imagen original BGR)
    aligned_face_bgr = align_face(img, landmarks)
    info(f"aligned_face_bgr shape : {aligned_face_bgr.shape}  (BGR, salida de align_face)")

    # Convertir a RGB antes de ArcFace
    # FIX: ArcFace w600k_mbf fue entrenado en RGB
    aligned_face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
    info(f"aligned_face_rgb shape : {aligned_face_rgb.shape}  (RGB, entrada a ArcFace)")

    embedding = inference_engine.get_face_embedding(aligned_face_rgb)

    if embedding.shape != (512,):
        fail(f"Embedding shape incorrecto: {embedding.shape}  (esperado: (512,))")
        return np.zeros(512)

    norm = float(np.linalg.norm(embedding))
    ok(f"Embedding extraido correctamente.")
    info(f"Shape : {embedding.shape}  (esperado: (512,))")
    info(f"Norma L2 : {norm:.6f}  (esperado: ~1.0 si ya fue normalizado)")
    info(f"Min : {embedding.min():.4f}  Max : {embedding.max():.4f}")

    # Test de similitud coseno del embedding consigo mismo (debe ser 1.0)
    from app.services.face_math import compute_cosine_similarity
    self_similarity = compute_cosine_similarity(embedding, embedding)
    if abs(self_similarity - 1.0) < 1e-5:
        ok(f"Similitud coseno consigo mismo: {self_similarity:.6f}  (esperado: 1.0)")
    else:
        warn(f"Similitud coseno consigo mismo: {self_similarity:.6f}  (deberia ser 1.0)")

    return embedding


# ------------------------------------------------------------------ #
#  7. EMOTION RECOGNITION (EmotiEffLib EfficientNet-B0)
#     FIX aplicado: aligned_face en RGB, keys alineadas al ORM
# ------------------------------------------------------------------ #
def test_emotion(aligned_face_rgb: np.ndarray) -> dict:
    section("PASO 7 - Emotion Recognition (EmotiEffLib EfficientNet-B0)")

    info(f"Input shape : {aligned_face_rgb.shape}  (RGB, sera resizado a 224x224 internamente)")

    emotion_result = inference_engine.detect_emotion(aligned_face_rgb)

    # Verificar que las keys coinciden con el ORM (dominant_emotion, confidence)
    # FIX: inference_engine.detect_emotion ahora retorna estas keys
    dominant  = emotion_result.get("dominant_emotion", "ERROR_KEY_NOT_FOUND")
    confidence = emotion_result.get("confidence", -1.0)

    if dominant == "ERROR_KEY_NOT_FOUND":
        fail("La key 'dominant_emotion' no existe en el resultado.")
        fail(f"Keys actuales: {list(emotion_result.keys())}")
        fail("Verifica el fix en inference_engine.detect_emotion")
        return emotion_result

    ok(f"Emocion detectada : {dominant}")
    ok(f"Confianza         : {confidence:.4f}")
    info(f"Resultado completo: {emotion_result}")

    EMOTION_CLASSES = [
        "Anger", "Contempt", "Disgust", "Fear",
        "Happiness", "Neutral", "Sadness", "Surprise"
    ]

    if dominant in EMOTION_CLASSES:
        ok(f"'{dominant}' es una clase valida del modelo.")
    else:
        warn(f"'{dominant}' no esta en las 8 clases esperadas: {EMOTION_CLASSES}")

    return emotion_result


# ------------------------------------------------------------------ #
#  8. TEST DE SIMILITUD ENTRE DOS EMBEDDINGS
#     Simula la comparacion biometrica que hace stream.py
# ------------------------------------------------------------------ #
def test_biometric_similarity(embedding: np.ndarray) -> None:
    section("PASO 8 - Simulacion de similitud biometrica (coseno)")

    from app.services.face_math import verify_biometric_match

    # Simular embedding almacenado en DB con ruido minimo
    noisy_embedding = embedding + np.random.normal(0, 0.01, embedding.shape).astype(np.float32)
    noisy_norm = np.linalg.norm(noisy_embedding)
    if noisy_norm > 0:
        noisy_embedding = noisy_embedding / noisy_norm

    is_match, similarity = verify_biometric_match(embedding, noisy_embedding, threshold=0.50)

    info(f"Embedding original  vs  embedding con ruido minimo:")
    info(f"Similitud coseno : {similarity:.4f}")
    info(f"Threshold        : 0.50")

    if is_match:
        ok(f"Match correcto  (similarity={similarity:.4f} >= 0.50)")
    else:
        warn(f"No match  (similarity={similarity:.4f} < 0.50)")
        warn("Esto puede indicar un problema en la normalizacion del embedding.")

    # Simular embedding de persona diferente (vector aleatorio normalizado)
    random_embedding = np.random.randn(512).astype(np.float32)
    random_embedding = random_embedding / np.linalg.norm(random_embedding)

    is_match_random, similarity_random = verify_biometric_match(embedding, random_embedding, threshold=0.50)

    info(f"\nEmbedding original  vs  embedding aleatorio (persona diferente):")
    info(f"Similitud coseno : {similarity_random:.4f}")

    if not is_match_random:
        ok(f"Rechazo correcto de persona diferente  (similarity={similarity_random:.4f} < 0.50)")
    else:
        warn(f"Falso positivo con vector aleatorio  (similarity={similarity_random:.4f})")
        warn("Considera subir el threshold en face_math.verify_biometric_match")


# ------------------------------------------------------------------ #
#  RUNNER PRINCIPAL
# ------------------------------------------------------------------ #
def run_smoke_test(image_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  FACE PIPELINE - MODEL SMOKE TEST")
    print(f"  Backend: FastAPI + ONNX Runtime")
    print(f"  Modelos: SCRFD / MiniFASNetV2 / ArcFace / EmotiEffLib")
    print(f"{'='*60}")

    # Pasos secuenciales del pipeline
    img = load_image(image_path)
    load_models()
    inspect_model_shapes()

    faces = test_detection(img)
    if not faces:
        warn("Sin rostros detectados. No es posible continuar con el pipeline.")
        print(f"\n{'='*60}")
        print("  SMOKE TEST FINALIZADO CON ADVERTENCIAS")
        print(f"{'='*60}\n")
        return

    primary_face = faces[0]
    bbox         = primary_face["bbox"]
    landmarks    = primary_face["landmarks"]

    liveness_score = test_liveness(img, bbox)

    # align_face retorna BGR, convertimos a RGB para ArcFace y EmotiEffLib
    from app.utils.image_processing import align_face as _align
    aligned_bgr = _align(img, landmarks)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    embedding = test_recognition(img, landmarks)

    test_emotion(aligned_rgb)

    if embedding is not None and embedding.shape == (512,):
        test_biometric_similarity(embedding)

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST FINALIZADO")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Puedes pasar una imagen como argumento o usar el default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "./faceEmotions/fear1.jpg"

    run_smoke_test(image_path)