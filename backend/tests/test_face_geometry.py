import pytest
import numpy as np

from app.services.face_geometry import (
    compute_ear_from_landmarks,
    classify_eye_state,
    compute_mar_from_landmarks,
    estimate_head_pose,
    analyze_face_geometry,
    EAR_BLINK_THRESHOLD,
    EAR_DROWSINESS_THRESHOLD,
)


# EAR TEST -CLASSIFY_EYE_STATE

class TestComputeEar:
    """Tests for Eye Aspect Ratio calculation."""

    def test_returns_float(self, frontal_landmarks_640x480):
        """Return type must be Python float for JSON serialization."""
        result = compute_ear_from_landmarks(frontal_landmarks_640x480)
        assert isinstance(result, float)

    def test_frontal_face_returns_positive_ear(self, frontal_landmarks_640x480):
        """
        A frontal face with open eyes must return a positive EAR.
        Negative EAR has no physical meaning.
        """
        result = compute_ear_from_landmarks(frontal_landmarks_640x480)
        assert result > 0.0, f"Expected positive EAR, got {result}"

    def test_zero_landmarks_returns_zero(self):
        """
        Landmarks at the origin produce zero inter-ocular distance.
        Must return 0.0 without raising ZeroDivisionError.
        """
        zero_landmarks = np.zeros((5, 2), dtype=np.float32)
        result = compute_ear_from_landmarks(zero_landmarks)
        assert result == 0.0

    def test_ear_is_rounded_to_4_decimals(self, frontal_landmarks_640x480):
        """
        EAR must be rounded to 4 decimal places as specified in the implementation.
        Ensures consistent output format for logging and WebSocket payloads.
        """
        result = compute_ear_from_landmarks(frontal_landmarks_640x480)
        assert result == round(result, 4)


# eye_state TEST -CLASSIFY_EYE_STATE
class TestClassifyEyeState:
    """Tests for eye state classification based on EAR value."""

    def test_open_eyes_above_blink_threshold(self):
        """
        EAR clearly above blink threshold must return eye_state='open'.
        """
        result = classify_eye_state(ear=0.35, consecutive_frames_below=0)
        assert result["eye_state"] == "open"
        assert result["is_blinking"] is False
        assert result["is_drowsy"] is False

    def test_blinking_below_blink_threshold(self):
        """
        EAR just below blink threshold with few frames must return 'blinking'.
        A single blink is not drowsiness.
        """
        result = classify_eye_state(
            ear=EAR_BLINK_THRESHOLD - 0.01,
            consecutive_frames_below=2
        )
        assert result["eye_state"] == "blinking"
        assert result["is_blinking"] is True
        assert result["is_drowsy"] is False

    def test_drowsy_after_sustained_low_ear(self):
        """
        EAR below drowsiness threshold sustained for 15+ frames
        must return eye_state='drowsy' and is_drowsy=True.
        This is the safety-critical detection for driver monitoring.
        """
        result = classify_eye_state(
            ear=EAR_DROWSINESS_THRESHOLD - 0.01,
            consecutive_frames_below=20
        )
        assert result["eye_state"] == "drowsy"
        assert result["is_drowsy"] is True

    def test_returns_ear_value_in_result(self, frontal_landmarks_640x480):
        """
        The result dict must include the original EAR value
        so the frontend can display it numerically.
        """
        ear = compute_ear_from_landmarks(frontal_landmarks_640x480)
        result = classify_eye_state(ear=ear)
        assert result["ear"] == ear

    def test_result_contains_required_keys(self):
        """
        Result dict must contain all keys expected by stream.py.
        """
        result = classify_eye_state(ear=0.30)
        assert "ear"         in result
        assert "eye_state"   in result
        assert "is_blinking" in result
        assert "is_drowsy"   in result



# compute_mar_from_landmarks()

class TestComputeMar:
    
    """Tests for Mouth Aspect Ratio calculation."""

    def test_returns_float(self, frontal_landmarks_640x480):
        """Return type must be Python float."""
        result = compute_mar_from_landmarks(frontal_landmarks_640x480)
        assert isinstance(result, float)

    def test_frontal_face_returns_positive_mar(self, frontal_landmarks_640x480):
        """Closed mouth on a frontal face must return a positive MAR."""
        result = compute_mar_from_landmarks(frontal_landmarks_640x480)
        assert result > 0.0

    def test_zero_landmarks_returns_zero(self):
        """
        Zero landmarks produce zero mouth width.
        Must return 0.0 without ZeroDivisionError.
        """
        zero_landmarks = np.zeros((5, 2), dtype=np.float32)
        result = compute_mar_from_landmarks(zero_landmarks)
        assert result == 0.0

    def test_wider_mouth_produces_lower_mar(self):
        """
        A wider mouth (larger horizontal distance between corners)
        with the same vertical distance produces a lower MAR.
        MAR = vertical / horizontal, so wider mouth = lower ratio.
        """
        narrow_mouth = np.array([
            [280.0, 200.0],
            [360.0, 200.0],
            [320.0, 240.0],
            [305.0, 280.0],   # narrow: 10px from center
            [335.0, 280.0],
        ], dtype=np.float32)

        wide_mouth = np.array([
            [280.0, 200.0],
            [360.0, 200.0],
            [320.0, 240.0],
            [270.0, 280.0],   # wide: 50px from center
            [370.0, 280.0],
        ], dtype=np.float32)

        mar_narrow = compute_mar_from_landmarks(narrow_mouth)
        mar_wide   = compute_mar_from_landmarks(wide_mouth)

        assert mar_narrow > mar_wide, (
            f"Narrower mouth should have higher MAR. "
            f"narrow={mar_narrow:.4f} wide={mar_wide:.4f}"
        )

# estimate_head_pose()

class TestEstimateHeadPose:
    """Tests for head pose estimation (pitch, yaw, roll)."""

    def test_returns_required_keys(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """
        Result dict must contain pitch, yaw, roll, pose_label, is_frontal.
        These keys are referenced in stream.py response_data.
        """
        result = estimate_head_pose(
            frontal_landmarks_640x480, image_width_640, image_height_480
        )
        assert "pitch"      in result
        assert "yaw"        in result
        assert "roll"       in result
        assert "pose_label" in result
        assert "is_frontal" in result

    def test_angles_are_floats(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """Pitch, yaw and roll must be Python floats for JSON serialization."""
        result = estimate_head_pose(
            frontal_landmarks_640x480, image_width_640, image_height_480
        )
        assert isinstance(result["pitch"], float)
        assert isinstance(result["yaw"],   float)
        assert isinstance(result["roll"],  float)

    def test_frontal_face_classified_as_frontal(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """
        Symmetric landmarks centered in a 640x480 frame must be
        classified as 'frontal'. Critical for the biometric enrollment
        endpoint which requires a frontal pose.
        """
        result = estimate_head_pose(
            frontal_landmarks_640x480, image_width_640, image_height_480
        )
        assert result["pose_label"] == "frontal"
        assert result["is_frontal"] is True

    def test_is_frontal_is_bool(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """is_frontal must be a bool, not a numpy bool or int."""
        result = estimate_head_pose(
            frontal_landmarks_640x480, image_width_640, image_height_480
        )
        assert isinstance(result["is_frontal"], bool)

    def test_valid_pose_labels(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """
        pose_label must be one of the 6 valid classifications
        defined in _classify_head_pose().
        """
        valid_labels = {
            "frontal", "looking_left", "looking_right",
            "looking_up", "looking_down", "tilted", "unknown"
        }
        result = estimate_head_pose(
            frontal_landmarks_640x480, image_width_640, image_height_480
        )
        assert result["pose_label"] in valid_labels


# analyze_face_geometry() - full pipeline 


class TestAnalyzeFaceGeometry:
    """Tests for the full geometry analysis pipeline."""

    def test_returns_three_top_level_keys(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """
        analyze_face_geometry() must return a dict with ear, mar, head_pose.
        This is the contract with stream.py which adds geometry to the
        WebSocket response payload.
        """
        result = analyze_face_geometry(
            frontal_landmarks_640x480, image_width_640, image_height_480
        )
        assert "ear"       in result
        assert "mar"       in result
        assert "head_pose" in result

    def test_does_not_raise_on_valid_input(
        self, frontal_landmarks_640x480, image_width_640, image_height_480
    ):
        """
        The full pipeline must complete without raising any exception
        for valid frontal landmarks. Robustness requirement for stream.py.
        """
        try:
            analyze_face_geometry(
                frontal_landmarks_640x480, image_width_640, image_height_480
            )
        except Exception as e:
            pytest.fail(f"analyze_face_geometry raised unexpectedly: {e}")

    def test_returns_fallback_on_invalid_landmarks(
        self, image_width_640, image_height_480
    ):
        """
        If landmarks are degenerate (all zeros), the function must return
        a safe fallback dict instead of crashing the WebSocket loop.
        This protects stream.py from unhandled exceptions mid-session.
        """
        bad_landmarks = np.zeros((5, 2), dtype=np.float32)
        result = analyze_face_geometry(
            bad_landmarks, image_width_640, image_height_480
        )
        assert "ear"       in result
        assert "mar"       in result
        assert "head_pose" in result
        
        
