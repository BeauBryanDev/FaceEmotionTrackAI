import pytest
import numpy as np
import cv2



# EMBEDDING FIXTURES
# Fixtures defined here are automatically available to all test files in the backend/tests/ directory.
@pytest.fixture
def unit_vector_512() -> np.ndarray:
    """
    Returns a reproducible L2-normalized 512D vector.
    Identical calls produce the same vector (seeded RNG).
    Simulates a valid ArcFace embedding stored in the database.
    """
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)



@pytest.fixture
def similar_vector_512(unit_vector_512) -> np.ndarray:
    """
    Returns a vector close to unit_vector_512 with small Gaussian noise.
    Simulates a live embedding from the same person under slightly
    different lighting or angle conditions.
    Expected cosine similarity vs unit_vector_512: ~0.95 - 0.99.
    """
    rng = np.random.default_rng(seed=99)
    noise = rng.standard_normal(512).astype(np.float32) * 0.01
    noisy = unit_vector_512 + noise
    return noisy / np.linalg.norm(noisy)


@pytest.fixture
def different_vector_512() -> np.ndarray:
    """
    Returns a random unit vector seeded differently from unit_vector_512.
    Simulates an embedding from a completely different person.
    Expected cosine similarity vs unit_vector_512: ~0.0 +/- 0.1.
    """
    rng = np.random.default_rng(seed=777)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def zero_vector_512() -> np.ndarray:
    """
    Returns a zero vector of 512 dimensions.
    Used to test division-by-zero guards in cosine similarity.
    """
    return np.zeros(512, dtype=np.float32)


@pytest.fixture
def embedding_matrix_10x512() -> np.ndarray:
    """
    Returns a (10, 512) matrix of L2-normalized embeddings.
    Used to test PCA reduction and batch operations.
    10 samples is above the MIN_SAMPLES_REQUIRED=3 threshold.
    """
    rng = np.random.default_rng(seed=2024)
    matrix = rng.standard_normal((10, 512)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


@pytest.fixture
def embedding_matrix_3x512() -> np.ndarray:
    """
    Returns a (3, 512) matrix - the minimum required for PCA.
    Tests boundary conditions.
    """
    rng = np.random.default_rng(seed=2025)
    matrix = rng.standard_normal((3, 512)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


@pytest.fixture
def embedding_matrix_2x512() -> np.ndarray:
    """
    Returns a (2, 512) matrix - BELOW the minimum required for PCA.
    Used to test that PCA raises ValueError with insufficient data.
    """
    rng = np.random.default_rng(seed=2026)
    matrix = rng.standard_normal((2, 512)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


# LANDMARK FIXTURES
# Point order: [left_eye, right_eye, nose, mouth_left, mouth_right]

@pytest.fixture
def frontal_landmarks_640x480() -> np.ndarray:
    """
    Realistic 5-point landmarks for a frontal face centered in a 640x480 frame.
    Used to test EAR, MAR, and Head Pose with expected frontal output.
    """
    return np.array([
        [280.0, 200.0],   # left eye
        [360.0, 200.0],   # right eye
        [320.0, 240.0],   # nose
        [295.0, 280.0],   # mouth left
        [345.0, 280.0],   # mouth right
    ], dtype=np.float32)


@pytest.fixture
def left_yaw_landmarks_640x480() -> np.ndarray:
    """
    Landmarks simulating head turned to the left (negative yaw).
    The right eye appears closer to the nose in 2D projection.
    Used to test that head pose classifier returns 'looking_left'.
    """
    return np.array([
        [250.0, 200.0],   # left eye - shifted further left
        [310.0, 205.0],   # right eye - closer to center
        [290.0, 240.0],   # nose - shifted left
        [265.0, 280.0],   # mouth left
        [310.0, 282.0],   # mouth right
    ], dtype=np.float32)



@pytest.fixture
def right_yaw_landmarks_640x480() -> np.ndarray:
    """
    Landmarks simulating head turned to the right (positive yaw).
    Used to test that head pose classifier returns 'looking_right'.
    """
    return np.array([
        [330.0, 205.0],   # left eye - closer to center
        [390.0, 200.0],   # right eye - shifted further right
        [350.0, 240.0],   # nose - shifted right
        [335.0, 282.0],   # mouth left
        [375.0, 280.0],   # mouth right
    ], dtype=np.float32)


@pytest.fixture
def image_width_640() -> int:
    """Standard webcam frame width."""
    return 640


@pytest.fixture
def image_height_480() -> int:
    """Standard webcam frame height."""
    return 480


# IMAGE FIXTURES

@pytest.fixture
def black_image_640x480() -> np.ndarray:
    """
    Returns a black BGR image of shape (480, 640, 3).
    Used as a minimal valid input for resize and tensor conversion tests.
    """
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_image_112x112() -> np.ndarray:
    """
    Returns a white BGR image of shape (112, 112, 3).
    Simulates an already-cropped and aligned face ready for ArcFace.
    """
    return np.full((112, 112, 3), 255, dtype=np.uint8)


@pytest.fixture
def random_face_image_480x480() -> np.ndarray:
    """
    Returns a random BGR image of shape (480, 480, 3).
    Simulates a raw webcam crop around a detected face region.
    """
    rng = np.random.default_rng(seed=123)
    return rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)


@pytest.fixture
def valid_base64_black_image() -> str:
    """
    Returns a valid Base64-encoded JPEG of a 100x100 black image.
    Used to test decode_base64_image with a real encoded payload.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    import base64
    return base64.b64encode(buffer).decode("utf-8")


@pytest.fixture
def valid_base64_with_data_uri(valid_base64_black_image) -> str:
    """
    Returns the same Base64 image prefixed with a data URI header.
    Tests that decode_base64_image correctly strips the header.
    """
    return f"data:image/jpeg;base64,{valid_base64_black_image}"


@pytest.fixture
def invalid_base64_string() -> str:
    """
    Returns a clearly invalid Base64 string.
    Tests that decode_base64_image returns None gracefully.
    """
    return "this_is_not_valid_base64_@@@@"

