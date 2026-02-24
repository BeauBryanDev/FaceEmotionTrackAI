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
        
        
    def detect_faces(self, image: np.ndarray, threshold: float = 0.5) -> list:
            """
            Executes the SCRFD model to detect faces in a given image.
            
            Args:
                image (np.ndarray): The raw BGR image captured from the webcam.
                threshold (float): The minimum confidence score to consider a valid face.
                
            Returns:
                list: A list of dictionaries, each containing 'bbox', 'score', and 'landmarks'
                    for every detected face.
            """
            session = self.get_session("detection")
            if not session:
                raise RuntimeError("Detection model session is not initialized.")

            # SCRFD typically operates optimally on specific input resolutions like 640x640.
            # We use the utility functions built previously to prepare the tensor.
            from app.utils.image_processing import convert_and_resize, prepare_tensor_for_onnx
            
            input_size = (640, 640)
            original_height, original_width = image.shape[:2]
            
            # Resize and convert BGR to RGB
            resized_image = convert_and_resize(image, input_size, to_rgb=True)
            
            # InsightFace SCRFD standard normalization: (pixel_value - 127.5) / 128.0
            # This centers the data around 0 with a standard deviation of 1.
            input_tensor = prepare_tensor_for_onnx(
                resized_image, 
                mean= 127.5 / 255.0, 
                std= 128.0 / 255.0
            )
            
            # Execute the ONNX runtime session
            input_name = session.get_inputs()[0].name
            raw_outputs = session.run(None, {input_name: input_tensor})
            
            # SCRFD outputs multiple tensors representing bounding box regressions, 
            # classification scores, and landmark predictions across different stride levels.
            # In a full production environment, these raw outputs are decoded using 
            # anchor generation and Non-Maximum Suppression (NMS).
            
            faces = self._decode_scrfd_outputs(
                raw_outputs, 
                threshold, 
                input_size, 
                (original_height, original_width)
            )
            
            return faces

    
    # def _decode_scrfd_outputs(self, raw_outputs: list, threshold: float, input_size: tuple, original_size: tuple) -> list:
    #     # SCRFD Outputs: [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
    #     # Nota: El orden puede variar según la exportación del ONNX. 
    #     # Esta lógica asume el formato estándar de InsightFace.
        
    #     scores_list = raw_outputs[0:3]
    #     bboxes_list = raw_outputs[3:6]
    #     kps_list = raw_outputs[6:9]
        
    #     total_faces = []
    #     strides = [8, 16, 32]
        
    #     for i, stride in enumerate(strides):
    #         scores = scores_list[i]
    #         bboxes = bboxes_list[i] * stride
    #         kps = kps_list[i] * stride
            
    #         height, width = input_size[0] // stride, input_size[1] // stride
            
    #         # Generar anchors (puntos de rejilla) para este stride
    #         anchor_grid = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
    #         anchor_grid = (anchor_grid * stride).reshape((-1, 2))
            
    #         # Filtrar por threshold
    #         pos_indices = np.where(scores.flatten() > threshold)[0]
            
    #         for idx in pos_indices:
    #             conf = scores.flatten()[idx]
    #             anchor = anchor_grid[idx]
                
    #             # Decodificar Bounding Box (Distancias desde el anchor)
    #             reg_bbox = bboxes.reshape((-1, 4))[idx]
    #             xmin = anchor[0] - reg_bbox[0]
    #             ymin = anchor[1] - reg_bbox[1]
    #             xmax = anchor[0] + reg_bbox[2]
    #             ymax = anchor[1] + reg_bbox[3]
                
    #             # Decodificar Landmarks (5 puntos)
    #             reg_kps = kps.reshape((-1, 10))[idx]
    #             landmarks = []
    #             for k in range(0, 10, 2):
    #                 px = anchor[0] + reg_kps[k]
    #                 py = anchor[1] + reg_kps[k+1]
    #                 landmarks.append([px, py])
                
    #             total_faces.append({
    #                 "bbox": [xmin, ymin, xmax, ymax],
    #                 "score": float(conf),
    #                 "landmarks": np.array(landmarks)
    #             })

    #     # Aplicar NMS (Non-Maximum Suppression) para evitar cajas duplicadas
    #     return self._apply_nms(total_faces, iou_threshold=0.4, original_size=original_size, input_size=input_size)
    
    
    def _decode_scrfd_outputs(self, raw_outputs: list, threshold: float, input_size: tuple, original_size: tuple) -> list:
        # El orden confirmado de los tensores es:
        # [0:3] -> Scores (12800, 3200, 800)
        # [3:6] -> BBoxes (12800x4, 3200x4, 800x4)
        # [6:9] -> KPS / Landmarks (12800x10, 3200x10, 800x10)
        
        scores_list = raw_outputs[0:3]
        bboxes_list = raw_outputs[3:6]
        kps_list = raw_outputs[6:9]
        
        total_faces = []
        strides = [8, 16, 32]
        
        for i, stride in enumerate(strides):
            scores = scores_list[i]
            bboxes = bboxes_list[i] * stride
            kps = kps_list[i] * stride
            
            height = input_size[0] // stride
            width = input_size[1] // stride
            
            # EL DETALLE CLAVE: Calcular cuántos anchors por píxel usa el modelo (12800 / 6400 = 2)
            num_anchors = scores.shape[0] // (height * width)
            
            # Generar rejilla base (X, Y)
            X, Y = np.meshgrid(np.arange(width), np.arange(height))
            anchor_grid = np.stack([X, Y], axis=-1)
            anchor_grid = (anchor_grid * stride).reshape((-1, 2))
            
            # Repetir la rejilla para que coincida con los 12800/3200/800 tensores
            anchor_grid = np.repeat(anchor_grid, num_anchors, axis=0)
            
            # Filtrar por threshold
            scores_flat = scores.flatten()
            pos_indices = np.where(scores_flat > threshold)[0]
            
            for idx in pos_indices:
                conf = scores_flat[idx]
                anchor = anchor_grid[idx]
                
                # Decodificar Bounding Box (xmin, ymin, xmax, ymax)
                reg_bbox = bboxes.reshape((-1, 4))[idx]
                xmin = anchor[0] - reg_bbox[0]
                ymin = anchor[1] - reg_bbox[1]
                xmax = anchor[0] + reg_bbox[2]
                ymax = anchor[1] + reg_bbox[3]
                
                # Decodificar Landmarks (5 puntos = 10 coordenadas)
                reg_kps = kps.reshape((-1, 10))[idx]
                landmarks = []
                for k in range(0, 10, 2):
                    px = anchor[0] + reg_kps[k]
                    py = anchor[1] + reg_kps[k+1]
                    landmarks.append([px, py])
                
                total_faces.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "score": float(conf),
                    "landmarks": np.array(landmarks)
                })

        return self._apply_nms(total_faces, iou_threshold=0.4, original_size=original_size, input_size=input_size)
    

    def _apply_nms(self, faces, iou_threshold, original_size, input_size):
        if not faces: return []
        
        # Ordenar por score
        faces.sort(key=lambda x: x['score'], reverse=True)
        keep = []
        
        while faces:
            best_face = faces.pop(0)
            keep.append(best_face)
            faces = [f for f in faces if self._compute_iou(best_face['bbox'], f['bbox']) < iou_threshold]
            
        # Escalar coordenadas de vuelta al tamaño original
        h_ratio = original_size[0] / input_size[0]
        w_ratio = original_size[1] / input_size[1]
        
        for f in keep:
            f['bbox'] = [f['bbox'][0]*w_ratio, f['bbox'][1]*h_ratio, f['bbox'][2]*w_ratio, f['bbox'][3]*h_ratio]
            f['landmarks'][:, 0] *= w_ratio
            f['landmarks'][:, 1] *= h_ratio
            
        return keep

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)    
    
    
    def get_face_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts the 512-dimensional biometric embedding using ArcFace.
        
        Args:
            aligned_face (np.ndarray): The 112x112 aligned RGB face image.
            
        Returns:
            np.ndarray: A 1D array of shape (512,) representing the L2-normalized face embedding.
        """
        session = self.get_session("recognition")
        if not session:
            raise RuntimeError("Recognition model session is not initialized.")

        # ArcFace w600k_mbf requires exactly 112x112 input and standard normalization
        from app.utils.image_processing import prepare_tensor_for_onnx
        
        input_tensor = prepare_tensor_for_onnx(
            aligned_face, 
            mean=0.5, 
            std=0.5
        )
        
        input_name = session.get_inputs()[0].name
        raw_outputs = session.run(None, {input_name: input_tensor})
        
        # The output is a batch of embeddings, shape (1, 512)
        # We extract the first element and flatten it to a 1D array
        embedding = raw_outputs[0].flatten()
        
        # L2 Normalization ensures the vector sits on a unit hypersphere, 
        # which is a strict mathematical requirement for Cosine Similarity.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    def check_liveness(self, face_crop: np.ndarray) -> float:
        """
        Evaluates the face crop to determine if it is a real person or a spoof attempt.
        
        Args:
            face_crop (np.ndarray): The cropped RGB face image.
            
        Returns:
            float: A confidence score between 0.0 and 1.0, where 1.0 is a verified live person.
        """
        session = self.get_session("liveness")
        if not session:
            raise RuntimeError("Liveness model session is not initialized.")

        from app.utils.image_processing import convert_and_resize, prepare_tensor_for_onnx
        
        # MiniFASNetV2 operates on an 80x80 input resolution
        resized_image = convert_and_resize(face_crop, (80, 80), to_rgb=False)
        
        input_tensor = prepare_tensor_for_onnx(
            resized_image, 
            mean=0.5, 
            std=0.5
        )
        
        input_name = session.get_inputs()[0].name
        raw_outputs = session.run(None, {input_name: input_tensor})
        
        # The model outputs logits for two classes: [Fake, Real]
        # We apply the Softmax function to convert these raw logits into probabilities
        logits = raw_outputs[0].flatten()
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Index 1 represents the "Real/Live" class
        liveness_score = float(probabilities[1])
        return liveness_score

    def detect_emotion(self, aligned_face: np.ndarray) -> dict:
        """
        Classifies the dominant facial emotion using the EfficientNet-B0 model.
        
        Args:
            aligned_face (np.ndarray): The aligned RGB face image.
            
        Returns:
            dict: A dictionary containing the 'emotion_label' and 'confidence_score'.
        """
        session = self.get_session("emotion")
        if not session:
            raise RuntimeError("Emotion model session is not initialized.")

        from app.utils.image_processing import convert_and_resize
        
        # HSEmotion models require 224x224 input
        resized_image = convert_and_resize(aligned_face, (224, 224), to_rgb=False)
        
        # EfficientNet expects standard ImageNet normalization, which uses specific 
        # arrays for mean and standard deviation across the RGB channels.
        image_float = resized_image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized_img = (image_float - mean) / std
        chw_image = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(chw_image, axis=0)
        
        input_name = session.get_inputs()[0].name
        raw_outputs = session.run(None, {input_name: input_tensor})
        
        logits = raw_outputs[0].flatten()
        
        # Softmax for probability distribution
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # The 8 classes defined by the AffectNet dataset standard
        emotion_classes = [
            "Anger", "Contempt", "Disgust", "Fear", 
            "Happiness", "Neutral", "Sadness", "Surprise"
        ]
        
        max_index = int(np.argmax(probabilities))
        
        return {
            "emotion_label": emotion_classes[max_index],
            "confidence_score": float(probabilities[max_index])
        }
        
        

# Global instance for the Singleton pattern
inference_engine = InferenceEngine()



