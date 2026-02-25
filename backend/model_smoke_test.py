import cv2
import numpy as np
import os
from app.services.inference_engine import inference_engine
from app.utils.image_processing import align_face

def run_smoke_test(image_path: str):
    print(f"--- Iniciando Smoke Test de Modelos AI ---")
    
    # 1. Cargar imagen
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra la imagen en {image_path}")
        return
    
    img = cv2.imread(image_path)
    print(f"Imagen cargada: {img.shape} (H, W, C)")

    # 2. Inicializar Motores (Carga ONNX)
    inference_engine.load_models()

    # 3. Test de Detección (SCRFD)
    print("\n[Testing SCRFD Detection...]")
    # Simulamos el pipeline de detect_faces para ver los shapes
    session = inference_engine.get_session("detection")
    # Imprimimos info de entrada del modelo real
    input_info = session.get_inputs()[0]
    print(f"Input esperado por SCRFD: {input_info.name} - {input_info.shape} - {input_info.type}")
    
    print("\n--- Analizando estructura de salida del ONNX ---")
    for idx, out in enumerate(session.get_outputs()):
        print(f"Output {idx}: name='{out.name}', shape={out.shape}")
    print("------------------------------------------------\n")
    
    faces = inference_engine.detect_faces(img)
    print(f"Rostros detectados: {len(faces)}")
    
    if len(faces) > 0:
        face = faces[0]
        landmarks = face['landmarks']
        bbox = face['bbox']
        
        # 4. Test de Alineación y Reconocimiento (ArcFace)
        print("\n[Testing ArcFace Alignment & Recognition...]")
        aligned = align_face(img, landmarks)
        print(f"Shape tras alineación afín: {aligned.shape}")
        
        embedding = inference_engine.get_face_embedding(aligned)
        print(f"Embedding extraído: {embedding.shape} (Esperado: 512,)")
        
        # 5. Test de Liveness (MiniFASNet)
        print("\n[Testing MiniFASNet Liveness...]")
        x1, y1, x2, y2 = map(int, bbox)
        crop = img[max(0,y1):y2, max(0,x1):x2]
        liveness_score = inference_engine.check_liveness(crop)
        print(f"Score de Liveness: {liveness_score:.4f}")

        # 6. Test de Emoción (EfficientNet)
        print("\n[Testing HSEmotion...]")
        emotion = inference_engine.detect_emotion(aligned)
        print(f"Resultado Emoción: {emotion}")
        
        # Script de verificacion del orden de clases MiniFASNetV2
 
        score = inference_engine.check_liveness(img)
        print(f"Score con foto real: {score}")
        # Si score > 0.5 -> index 1 = Real (correcto, no cambiar nada)
        # Si score < 0.5 -> index 1 = Fake (invertir: liveness_score = float(probabilities[0]))

    print("\n--- Smoke Test Finalizado ---")

if __name__ == "__main__":
    # Asegúrate de poner una imagen de prueba en backend/test_face.jpg
    #run_smoke_test("test_face2.jpg")
    run_smoke_test("./faceEmotions/test_face.jpg")
