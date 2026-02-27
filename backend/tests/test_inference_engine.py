import pytest
import numpy as np
from unittest.mock import patch, MagicMock

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
                
                