import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from app.services.inference_engine import InferenceEngine
from tests.mocks import (
    make_mock_detection_session,
    make_mock_recognition_session,
    make_mock_liveness_session,
    make_mock_emotion_session,
)

# All tests are pure unit tests with no database, no ONNX, no HTTP.
# They are meant to be run in a clean environment with no external dependencies.

EMOTION_CLASSES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise"
]

def _make_engine_with_mock_sessions() -> InferenceEngine:
    """
    Returns an InferenceEngine whose sessions dict is populated
    with mock ONNX sessions. No real .onnx files are touched.
    """
    engine = InferenceEngine()
    engine.sessions = {
        "detection"  : make_mock_detection_session(),
        "recognition": make_mock_recognition_session(),
        "liveness"   : make_mock_liveness_session(real_score=0.92),
        "emotion"    : make_mock_emotion_session(dominant_class_index=4),
    }
    return engine


# Load Models() method

class TestLoadModels:
    """Tests for InferenceEngine.load_models()."""
    
    @patch("app.services.inference_engine.ort.InferenceSession")
    def test_loads_all_models(self, mock_ort_session):
        """
        load_models() loads all models from the models/ directory.
        """
        engine = _make_engine_with_mock_sessions()
        engine.load_models()
        
        assert len(engine.sessions) == 4
        assert "detection" in engine.sessions
        assert "recognition" in engine.sessions
        assert "liveness" in engine.sessions
        assert "emotion" in engine.sessions
        
    
    @patch("os.path.exists", return_value=True)
    @patch("onnxruntime.InferenceSession")
    def test_load_models_success(self, mock_session, mock_exists):
        mock_session.return_value = MagicMock()
        engine = InferenceEngine()
        engine.load_models()
        assert len(engine.sessions) == 4
        
    
    def test_load_models_success_when_all_files_exist(self):
        """
        load_models() must populate self.sessions with 4 entries
        when all model files exist on disk.
        We patch os.path.exists to return True and
        ort.InferenceSession to avoid loading real files.
        """
        engine = InferenceEngine()

        with patch("os.path.exists", return_value=True), \
             patch("onnxruntime.InferenceSession") as mock_session_cls:

            mock_session_cls.return_value = MagicMock()
            engine.load_models()

        assert len(engine.sessions) == 4
        assert "detection"   in engine.sessions
        assert "recognition" in engine.sessions
        assert "liveness"    in engine.sessions
        assert "emotion"     in engine.sessions


    def test_load_models_raises_when_file_missing(self):
        """
        load_models() must raise FileNotFoundError immediately
        when any model file does not exist on disk.
        The error message must include the missing path.
        """
        engine = InferenceEngine()

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                engine.load_models()

        assert "not found" in str(exc_info.value).lower()
        
    
    def test_load_models_raises_on_onnx_runtime_error(self):
        """
        If ort.InferenceSession raises (corrupt file, wrong opset, etc.),
        load_models() must propagate the exception instead of silencing it.
        """
        engine = InferenceEngine()

        with patch("os.path.exists", return_value=True), \
             patch("onnxruntime.InferenceSession",
                   side_effect=RuntimeError("Invalid ONNX model")):

            with pytest.raises(RuntimeError):
                engine.load_models()
                


class TestClearModels:
    """Tests for InferenceEngine.clear_models()."""
    
    def test_clear_models_empties_sessions(self):
        """
        clear_models() must leave self.sessions as an empty dict.
        Simulates the FastAPI lifespan shutdown hook.
        """
        engine = _make_engine_with_mock_sessions()
        assert len(engine.sessions) == 4

        engine.clear_models()

        assert len(engine.sessions) == 0
        
        
    def test_clear_models_is_idempotent(self):
        """
        Calling clear_models() twice must not raise any exception.
        """
        engine = _make_engine_with_mock_sessions()
        engine.clear_models()
        engine.clear_models()
        assert engine.sessions == {}
        
        
# get_face_embedding()

class TestGetFaceEmbedding:
    """Tests for InferenceEngine.get_face_embedding()."""

    def test_get_face_embedding_returns_embedding(self):
        """
        get_face_embedding() must return the face embedding vector
        from the recognition model.
        """
        engine = _make_engine_with_mock_sessions()
        embedding = engine.get_face_embedding(np.zeros((1, 3)))
        
        assert embedding.shape == (1, 512)
        assert embedding.dtype == np.float32
        
        
    def test_returns_512_dimensional_vector(self, white_image_112x112):
        """
        get_face_embedding() must return a 1D numpy array of shape (512,).
        This is the contract that pgvector and face_math.py depend on.
        """
        engine = _make_engine_with_mock_sessions()
        embedding = engine.get_face_embedding(white_image_112x112)

        assert embedding.shape == (512,), (
            f"Expected (512,), got {embedding.shape}"
        )
        
    def test_embedding_is_l2_normalized(self, white_image_112x112):
        """
        The returned embedding must be L2-normalized (unit vector).
        Required for cosine similarity to work correctly in face_math.py
        and for pgvector distance operators.
        """
        engine = _make_engine_with_mock_sessions()
        embedding = engine.get_face_embedding(white_image_112x112)

        norm = float(np.linalg.norm(embedding))
        assert abs(norm - 1.0) < 1e-5, (
            f"Expected L2 norm ~1.0, got {norm}"
        )


    def test_returns_ndarray(self, white_image_112x112):
        """Return type must be numpy ndarray, not list or tensor."""
        engine = _make_engine_with_mock_sessions()
        embedding = engine.get_face_embedding(white_image_112x112)

        assert isinstance(embedding, np.ndarray)
        
        
    def test_raises_when_session_not_initialized(self, white_image_112x112):
        """
        If the recognition session was never loaded,
        get_face_embedding() must raise RuntimeError with a clear message.
        """
        engine = InferenceEngine()  # empty sessions

        with pytest.raises(RuntimeError) as exc_info:
            engine.get_face_embedding(white_image_112x112)

        assert "recognition" in str(exc_info.value).lower()
        
    
    
# Check Livenesss

class TestCheckLiveness:
    """Tests for InferenceEngine.check_liveness()."""
    
    def test_returns_float(self, random_face_image_480x480):
        """
        check_liveness() must return a Python float.
        FastAPI WebSocket serialization requires native Python types.
        """
        engine = _make_engine_with_mock_sessions()
        score = engine.check_liveness(random_face_image_480x480)

        assert isinstance(score, float)
        
        
    def test_score_is_bounded_between_zero_and_one(
            self, random_face_image_480x480
        ):
            """
            Liveness score is a softmax probability.
            Must always be in [0.0, 1.0].
            """
            engine = _make_engine_with_mock_sessions()
            score = engine.check_liveness(random_face_image_480x480)

            assert 0.0 <= score <= 1.0, (
                f"Liveness score out of bounds: {score}"
            )
        
        
    def test_high_score_for_live_person(self, random_face_image_480x480):
        """
        When the mock session is configured to return a high liveness score,
        the function must reflect that value above the operational threshold.
        """
        engine = InferenceEngine()
        engine.sessions = {
            "liveness": make_mock_liveness_session(real_score=0.92)
        }
        score = engine.check_liveness(random_face_image_480x480)

        assert score > 0.60, (
            f"Expected score above stream threshold 0.60, got {score}"
        )
        
        
    def test_low_score_for_spoof_attempt(self, random_face_image_480x480):
        """
        When the mock returns a low liveness score (spoof attack),
        the value must be below the operational threshold.
        """
        engine = InferenceEngine()
        engine.sessions = {
            "liveness": make_mock_liveness_session(real_score=0.10)
        }
        score = engine.check_liveness(random_face_image_480x480)

        assert score < 0.60, (
            f"Expected score below threshold 0.60 for spoof, got {score}"
        )
        
    
    def test_raises_when_session_not_initialized(
        self, random_face_image_480x480
    ):
        """
        Must raise RuntimeError if liveness session was never loaded.
        """
        engine = InferenceEngine()

        with pytest.raises(RuntimeError) as exc_info:
            engine.check_liveness(random_face_image_480x480)

        assert "liveness" in str(exc_info.value).lower()
        
        
# Check Emotion

class TestDetectEmotion:
    """Tests for InferenceEngine.detect_emotion()."""
    
    def test_returns_dict_with_required_keys(self, white_image_112x112):
        """
        detect_emotion() must return a dict with exactly these keys:
        dominant_emotion, confidence, emotion_scores.
        These keys are referenced in stream.py and emotions router.
        """
        engine = _make_engine_with_mock_sessions()
        result = engine.detect_emotion(white_image_112x112)

        assert "dominant_emotion" in result
        assert "confidence"       in result
        assert "emotion_scores"   in result
        
        
    def test_dominant_emotion_is_valid_class(self, white_image_112x112):
        """
        The dominant_emotion value must be one of the 8 AffectNet classes.
        """
        engine = _make_engine_with_mock_sessions()
        result = engine.detect_emotion(white_image_112x112)

        assert result["dominant_emotion"] in EMOTION_CLASSES, (
            f"Got unexpected emotion label: {result['dominant_emotion']}"
        )
        
        
    def test_confidence_is_bounded(self, white_image_112x112):
        """
        Confidence is a softmax probability. Must be in [0.0, 1.0].
        """
        engine = _make_engine_with_mock_sessions()
        result = engine.detect_emotion(white_image_112x112)

        assert 0.0 <= result["confidence"] <= 1.0
        
        
    def test_emotion_scores_contains_all_8_classes(self, white_image_112x112):
        """
        emotion_scores must contain exactly 8 entries, one per class.
        Used by the frontend analytics dashboard and stored in JSONB.
        """
        engine = _make_engine_with_mock_sessions()
        result = engine.detect_emotion(white_image_112x112)

        assert len(result["emotion_scores"]) == 8
        for label in EMOTION_CLASSES:
            assert label in result["emotion_scores"], (
                f"Missing emotion class in scores: {label}"
            )
            
            
    def test_mock_returns_happiness_as_dominant(self, white_image_112x112):
        """
        The mock is configured with dominant_class_index=4 (Happiness).
        Verifies the mock wiring is correct end to end.
        """
        engine = _make_engine_with_mock_sessions()
        result = engine.detect_emotion(white_image_112x112)

        assert result["dominant_emotion"] == "Happiness"
        
        
    def test_raises_when_session_not_initialized(self, white_image_112x112):
        """
        Must raise RuntimeError if emotion session was never loaded.
        """
        engine = InferenceEngine()

        with pytest.raises(RuntimeError) as exc_info:
            engine.detect_emotion(white_image_112x112)

        assert "emotion" in str(exc_info.value).lower()
        
        
    
