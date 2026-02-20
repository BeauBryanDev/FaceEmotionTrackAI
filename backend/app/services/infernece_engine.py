import onnxruntime as ort
import numpy as np
import os
from app.core.config import settings

class InferenceEngine:
    """
    Singleton class to manage ONNX models and inference sessions.
    Handles Face Detection, Liveness, Recognition, and Emotion analysis.
    """
    def __init__(self):
        self.sessions = {}
        self.model_paths = {
            "detection": os.path.join(settings.MODELS_PATH, "detection/det_500m.onnx"),
            "recognition": os.path.join(settings.MODELS_PATH, "recognition/w600k_mbf.onnx"),
            "liveness": os.path.join(settings.MODELS_PATH, "liveness/minifasnet_v2.onnx"),
            "emotion": os.path.join(settings.MODELS_PATH, "emotion/emotieff_b0.onnx")
        }

    def load_models(self):
        """
        Initializes ONNX Runtime sessions for all models.
        Optimized for CPU usage using CPUExecutionProvider.
        """
        providers = ['CPUExecutionProvider']
        
        try:
            for name, path in self.model_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found at {path}")
                
                # Initialize session and store it in the dictionary
                self.sessions[name] = ort.InferenceSession(path, providers=providers)
                print(f"Successfully loaded {name} model.")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise e

    def get_session(self, model_name: str):
        """
        Returns the specific inference session.
        """
        return self.sessions.get(model_name)

    def clear_models(self):
        """
        Clears sessions from memory during shutdown.
        """
        self.sessions.clear()

# Global instance for the Singleton pattern
inference_engine = InferenceEngine()
