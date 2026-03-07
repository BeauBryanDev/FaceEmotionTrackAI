import numpy as np

def calculate_russell_coordinates(probabilities):
    """
    Convierte vector de probabilidades (8 emociones) a coordenadas Russell (X, Y).
    Orden esperado: [Happiness, Surprise, Contempt, Neutral, Disgust, Fear, Anger, Sadness]
    """
    # 1. Definimos los pesos de Valencia (X) y Arousal (Y) para cada emoción
    # Estos valores están normalizados entre -1 y 1
    weights = {
        # Emoción: (Valencia, Arousal)
        "Happiness": (0.8,  0.6),
        "Surprise":  (0.3,  0.8),
        "Contempt":  (-0.5, 0.2),
        "Neutral":   (0.0,  0.0),
        "Disgust":   (-0.7, 0.2),
        "Fear":      (-0.6, 0.7),
        "Anger":     (-0.7, 0.8),
        "Sadness":   (-0.8, -0.6)
    }
    
    # Convertimos los pesos a matrices de numpy para cálculo rápido
    valencia_weights = np.array([v[0] for v in weights.values()])
    arousal_weights = np.array([v[1] for v in weights.values()])
    
    # 2. Cálculo del promedio ponderado (Dot Product)
    # Esto evita saltos bruscos y suaviza la serie de tiempo
    x_coord = np.dot(probabilities, valencia_weights)
    y_coord = np.dot(probabilities, arousal_weights)
    
    return round(x_coord, 4), round(y_coord, 4)

# --- EJEMPLO DE USO ---
# Supongamos que tu modelo ONNX devuelve estas probabilidades:
# [Hap, Sur, Con, Neu, Dis, Fea, Ang, Sad]
probs_ejemplo = [0.10, 0.05, 0.05, 0.70, 0.02, 0.03, 0.03, 0.02] 

v, a = calculate_russell_coordinates(probs_ejemplo)

print(f"Coordenadas en el Plano de Russell: Valencia={v}, Arousal={a}")
