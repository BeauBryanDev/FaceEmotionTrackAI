import numpy as np
from unittest.mock import MagicMock
from datetime import datetime, timezone


def make_mock_session(output_arrays: list) -> MagicMock:
    """
    Factory function that creates a generic mock ONNX session.

    Args:
        output_arrays: List of numpy arrays to return from session.run().
                       Must match the real model output shapes.

    Returns:
        MagicMock configured to behave like ort.InferenceSession.
    """
    session = MagicMock()

    # Mock the input descriptor - all our models use a single input
    mock_input = MagicMock()
    mock_input.name = "input"
    session.get_inputs.return_value = [mock_input]

    # session.run() returns the list of output arrays
    session.run.return_value = output_arrays

    return session


def make_mock_detection_session() -> MagicMock:
    """
    Mock for SCRFD face detection model.

    Real output structure (9 tensors):
        [0:3] -> scores  at strides 8, 16, 32  shapes: (12800,), (3200,), (800,)
        [3:6] -> bboxes  at strides 8, 16, 32  shapes: (12800,4), (3200,4), (800,4)
        [6:9] -> kps     at strides 8, 16, 32  shapes: (12800,10), (3200,10), (800,10)

    This mock returns all-zero tensors so _decode_scrfd_outputs produces
    zero detections (all scores below threshold). This is the expected
    behavior for a black frame with no face.
    """
    outputs = [
        # Scores - all zeros means no face detected
        np.zeros((12800,),    dtype=np.float32),
        np.zeros((3200,),     dtype=np.float32),
        np.zeros((800,),      dtype=np.float32),
        # BBoxes
        np.zeros((12800, 4),  dtype=np.float32),
        np.zeros((3200,  4),  dtype=np.float32),
        np.zeros((800,   4),  dtype=np.float32),
        # Keypoints
        np.zeros((12800, 10), dtype=np.float32),
        np.zeros((3200,  10), dtype=np.float32),
        np.zeros((800,   10), dtype=np.float32),
    ]
    return make_mock_session(outputs)


def make_mock_recognition_session(embedding_value: float = 0.5) -> MagicMock:
    """
    Mock for ArcFace MobileFaceNet recognition model.

    Real output: (1, 512) embedding vector.
    The mock returns a uniform vector that when L2-normalized
    produces a valid unit vector.

    Args:
        embedding_value: Fill value for all 512 dimensions.
                         Default 0.5 produces a valid non-zero vector.
    """
    raw_embedding = np.full((1, 512), embedding_value, dtype=np.float32)
    return make_mock_session([raw_embedding])


def make_mock_liveness_session(real_score: float = 0.92) -> MagicMock:
    """
    Mock for MiniFASNetV2 liveness detection model.

    Real output: (1, 2) logits for [Real, Fake] classes.
    We set logits so that after softmax, probabilities[1] == real_score.

    Args:
        real_score: Desired liveness score after softmax (0.0 to 1.0).
                    Default 0.92 simulates a clearly live person.
    """
    # Important! Math to convert desired real_score to logits:
    
    # If we want softmax([a, b])[1] = real_score:
    # real_score = exp(b) / (exp(a) + exp(b))
    # Setting a=0: real_score = exp(b) / (1 + exp(b))
    # b = log(real_score / (1 - real_score))
    import math
    
    epsilon = 1e-7
    real_score = max(epsilon, min(1.0 - epsilon, real_score))
    logit_real = math.log(real_score / (1.0 - real_score))
    logits = np.array([[0.0, logit_real]], dtype=np.float32)
    
    return make_mock_session([logits])


def make_mock_emotion_session(dominant_class_index: int = 4) -> MagicMock:
    """
    Mock for EmotiEffLib EfficientNet-B0 emotion model.

    Real output: (1, 8) logits for 8 emotion classes.
    emotion_classes = [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
    Index 4 = Happiness (default).

    Args:
        dominant_class_index: Index of the emotion class to make dominant.
                              0=Anger, 1=Contempt, 2=Disgust, 3=Fear,
                              4=Happiness, 5=Neutral, 6=Sadness, 7=Surprise
    """
    logits = np.zeros((1, 8), dtype=np.float32)
    # Set the dominant class logit high so softmax selects it clearly
    logits[0, dominant_class_index] = 10.0
    
    return make_mock_session([logits])


# DATABASE MODEL MOCKS
# Simulate SQLAlchemy ORM objects without a real database connection.
def mock_user(
    user_id: int = 1,
    full_name: str = "Test User",
    email: str = "test@example.com",
    is_active: bool = True,
    is_superuser: bool = False,
    face_embedding: list = None
) -> MagicMock:
    """
    Creates a mock User ORM object.

    Args:
        user_id:        Primary key of the user.
        full_name:      Display name.
        email:          Unique email address.
        is_active:      Account active status.
        is_superuser:   Superuser flag.
        face_embedding: Optional 512D embedding as Python list.
                        If None, simulates a user who has not enrolled biometrics.

    Returns:
        MagicMock configured to behave like a User SQLAlchemy model instance.
    """
    user = MagicMock()
    user.id             = user_id
    user.full_name      = full_name
    user.email          = email
    user.is_active      = is_active
    user.is_superuser   = is_superuser
    user.face_embedding = face_embedding
    user.created_at     = datetime.now(timezone.utc)
    user.updated_at     = None
    
    return user


def mock_user_with_embedding(user_id: int = 1) -> MagicMock:
    """
    Creates a mock User with a valid 512D L2-normalized face embedding.
    Simulates a fully enrolled user ready for biometric authentication.
    """
    rng = np.random.default_rng(seed=user_id)
    vec = rng.standard_normal(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    
    return mock_user(
        user_id=user_id,
        face_embedding=vec.tolist()
    )


def mock_emotion_record(
    record_id: int = 1,
    user_id: int = 1,
    dominant_emotion: str = "Happiness",
    confidence: float = 0.92,
    emotion_scores: dict = None
) -> MagicMock:
    """
    Creates a mock Emotion ORM record.
    Used in test_emotions.py to simulate DB query results.
    """
    record = MagicMock()
    record.id               = record_id
    record.user_id          = user_id
    record.dominant_emotion = dominant_emotion
    record.confidence       = confidence
    record.emotion_scores   = emotion_scores
    record.timestamp        = datetime.now(timezone.utc)
    
    return record