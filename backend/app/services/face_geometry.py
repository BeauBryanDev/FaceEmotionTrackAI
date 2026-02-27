import cv2
import numpy as np
from typing import Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)

# SCRFD 5-landmark indices
IDX_LEFT_EYE   = 0
IDX_RIGHT_EYE  = 1
IDX_NOSE       = 2
IDX_MOUTH_LEFT = 3
IDX_MOUTH_RIGHT= 4

# Thresholds for fatigue detection
EAR_BLINK_THRESHOLD      = 0.22  # Blink if EAR < 0.22
EAR_DROWSINESS_THRESHOLD = 0.18  # Drowsy if EAR < 0.18 for 15+ frames
DROWSINESS_FRAME_THRESHOLD = 15   # Drowsy if EAR < 0.18 for 15+ frames

# Reference: Guo et al. "Face Alignment by Explicit Shape Regression" (2012)
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),    # Nose (central reference point)
    (-30.0, -30.0, -30.0),   # Left eye
    (30.0,  -30.0, -30.0),   # Right eye
    (-25.0,  25.0, -25.0),   # Left mouth corner
    (25.0,   25.0, -25.0),   # Right mouth corner
], dtype=np.float64)

# Landmark indices used for the SolvePnP algorithm
SOLVEPNP_LANDMARK_INDICES = [
    IDX_NOSE,
    IDX_LEFT_EYE,
    IDX_RIGHT_EYE,
    IDX_MOUTH_LEFT,
    IDX_MOUTH_RIGHT
]

# EAR Eye Aspect Ratio
def compute_ear_from_landmarks(landmarks: np.ndarray) -> float :
    """
        Computes the ear shape from the given landmarks.
        Calculate the approximate Eye Aspect Ratio (EAR) using the two SCRFD eye landmarks.

        With only two points (left and right eye), we cannot calculate the complete 6-point Dlib EAR. Instead, I calculate the ratio of the estimated vertical inter-ocular distance to the horizontal distance.

        For a more accurate EAR, MediaPipe Face Mesh (468 points) is recommended. With SCRFD, we use an approximation that is sufficient for real-time drowsiness detection.

        Args:

        landmarks: array (5, 2) with the five SCRFD keypoints in the original image coordinates (already rescaled by _apply_nms).

        Returns:

        float: Approximate EAR. Normal values: 0.25–0.35.
        Blinking: < 0.22. Sustained drowsiness: < 0.18.
        
    """
    
# Extract individual coordinates for readability
    left_eye  = landmarks[IDX_LEFT_EYE]    # (x, y)
    right_eye = landmarks[IDX_RIGHT_EYE]   # (x, y)
    nose      = landmarks[IDX_NOSE]        # (x, y)
    mouth_l   = landmarks[IDX_MOUTH_LEFT]  # (x, y)
    mouth_r   = landmarks[IDX_MOUTH_RIGHT] # (x, y)

    
    # Inter-ocular horizontal distance (EAR denominator)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))

    if eye_distance < 1e-6:
        
        return 0.0
    
    eye_center = (left_eye + right_eye) / 2.0
    # Estimated vertical component: distance from eye center to nose.
    # This distance is normalized by the inter-ocular distance.
    # When eyes close, this component decreases as landmarks shift downwards.
    vertical_component = float(np.linalg.norm(eye_center - nose))
    print(f"Left eye: {left_eye}, Right eye: {right_eye}, Nose: {nose}")
    ear = (vertical_component / eye_distance)
    # Scale factor to align with standard EAR ranges
    ear_normalized = ear * 0.5
    # Scaled to match standard EAR ranges
    
    logger.debug(f"EAR calculation: vertical_component={vertical_component:.2f}, eye_distance={eye_distance:.2f}, ear={ear:.4f}, ear_normalized={ear_normalized:.4f}")
    
    if ear_normalized < 0.22:
        
        return 0.0
    
    else:
        
        return round( float(ear_normalized), 2)
    
# Classify eye state
def classify_eye_state(
    ear: float,
    consecutive_frames_below: int = 0
) -> dict:
    """
    Classifies the eye state based on EAR and temporal consistency.

    Determines if the subject is blinking, showing signs of drowsiness, or 
    has their eyes open/closed based on predefined thresholds.

    Args:
        ear (float): Eye Aspect Ratio calculated by `compute_ear_from_landmarks`.
        consecutive_frames_below (int): Number of consecutive frames where the 
            EAR has remained below the drowsiness threshold.

    Returns:
        dict con:
            eye_state: "open" | "blinking" | "drowsy" | "closed"
            is_blinking: bool : True if EAR is below blink threshold
            is_drowsy: bool : True if EAR is low for a long time
            ear: float : EAR value
    """
    
    is_blinking = ear < EAR_BLINK_THRESHOLD
    
    is_drowsy   = (
        ear < EAR_DROWSINESS_THRESHOLD and
        consecutive_frames_below >= DROWSINESS_FRAME_THRESHOLD
    )

    if is_drowsy:
        
        eye_state = "drowsy"
        
    elif ear < EAR_DROWSINESS_THRESHOLD:
        
        eye_state = "closed"
        
    elif is_blinking:
        
        eye_state = "blinking"
        
    else:
        
        eye_state = "open"
        
        
    print(f"EAR: {ear:.4f}, Blinking: {is_blinking}, Drowsy: {is_drowsy}, State: {eye_state}")

    return {
        "ear"        : ear,
        "eye_state"  : eye_state,
        "is_blinking": is_blinking,
        "is_drowsy"  : is_drowsy,
    }


# MAR - Mouth Aspect Ratio (deteccion de bostezo)
def compute_mar_from_landmarks(landmarks: np.ndarray) -> float:
    """
        Calculates the Mouth Aspect Ratio (MAR) using SCRFD mouth landmarks.

    Computes the ratio of the mouth's vertical opening to its horizontal width.
    Since SCRFD provides only the mouth corners (left and right) and not the 
    upper/lower lips, this method uses the distance between the nose and the 
    mouth center as a proxy for vertical opening.

        MAR = vertical_proxy_distance / horizontal_mouth_width

    Args:
        landmarks (np.ndarray): A (5, 2) array containing the SCRFD keypoints 
            in image coordinates.

    Returns:
        float: The approximate MAR value. Typically, a yawn is detected 
            if MAR > 0.6. Returns 0.0 if mouth width is negligible.
    """
    nose    = landmarks[IDX_NOSE]
    mouth_l = landmarks[IDX_MOUTH_LEFT]
    mouth_r = landmarks[IDX_MOUTH_RIGHT]

    # Horizontal mouth width (denominator)
    mouth_width = float(np.linalg.norm(mouth_r - mouth_l))

    if mouth_width < 1e-6:
        return 0.0

    # Midpoint between the two mouth corners
    mouth_center = (mouth_l + mouth_r) / 2.0

    # Vertical distance from nose to mouth center used as an opening proxy
    vertical = float(np.linalg.norm(nose - mouth_center))

    mar = vertical / mouth_width
    print(f"MAR calculation: vertical={vertical:.2f}, mouth_width={mouth_width:.2f}, mar={mar:.4f}")
    
    return round(float(mar), 4)


# HEAD POSE ESTIMATION 
def estimate_head_pose(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int
) -> dict:
    """
    Estimates head rotation angles (pitch, yaw, roll) using OpenCV's PnP algorithm.

    Applied Mathematics:
        - solvePnP: Solves the 3D-2D correspondence problem. Given a set of 3D 
          facial model points and their 2D projections, it finds the rotation (R) 
          and translation (T) vectors that minimize the reprojection error.
        - Rodrigues: Converts the rotation vector into a 3x3 rotation matrix.
        - Euler Angles: Extracted from the rotation matrix to determine orientation.

    Coordinate System (Orientation):
        - Pitch (+): Looking up    / (-): Looking down
        - Yaw   (+): Turning right / (-): Turning left
        - Roll  (+): Tilting right / (-): Tilting left

    Args:
        landmarks (np.ndarray): A (5, 2) array containing SCRFD keypoints.
        image_width (int): Width of the original frame in pixels.
        image_height (int): Height of the original frame in pixels.

    Returns:
        dict: Head pose data containing:
            - pitch (float): Up/down rotation in degrees.
            - yaw (float): Left/right rotation in degrees.
            - roll (float): Side tilt in degrees.
            - pose_label (str): Classification (e.g., "frontal", "tilted").
            - is_frontal (bool): True if the pose is within frontal thresholds.
            - rotation_vector (list): The raw rotation vector from solvePnP.
            
    """
    # 2D frame points ordered according to SOLVEPNP_LANDMARK_INDICES
    image_points_2d = np.array([
        
        landmarks[idx] for idx in SOLVEPNP_LANDMARK_INDICES
        
    ], dtype=np.float64)

    # Camera intrinsic matrix (approximation for standard webcams)
    # Common heuristic: focal length approximately equals image width
    focal_length = float(image_width)
    center       = (image_width / 2.0, image_height / 2.0)

    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)

    # Distortion coefficients (assuming no lens distortion)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP: find rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_3D_POINTS,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        logger.warning("cv2.solvePnP failed to converge. Returning neutral pose.")
        return {
            "pitch"          : 0.0,
            "yaw"            : 0.0,
            "roll"           : 0.0,
            "pose_label"     : "unknown",
            "is_frontal"     : False,
            "rotation_vector": None,
        }

    # Convert Rodrigues Rotation Vector to 3x3 Rotation Matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles from the rotation matrix
    pitch = float(np.degrees(np.arctan2(
        rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )))
    yaw = float(np.degrees(np.arctan2(
        -rotation_matrix[2, 0],
        np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)
    )))
    roll = float(np.degrees(np.arctan2(
        rotation_matrix[1, 0],
        rotation_matrix[0, 0]
    )))

    pitch = round(pitch, 2)
    yaw   = round(yaw,   2)
    roll  = round(roll,  2)

    pose_label = _classify_head_pose(yaw, pitch, roll)
    is_frontal = pose_label == "frontal"

    print(f"Head pose estimation: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}, pose_label={pose_label}, is_frontal={is_frontal}")
    
    # check this articlo for more infor : 
    # https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    logger.debug(f"Head pose estimation: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}, pose_label={pose_label}, is_frontal={is_frontal}")
    
    return {
        "pitch"          : pitch,
        "yaw"            : yaw,
        "roll"           : roll,
        "pose_label"     : pose_label,
        "is_frontal"     : is_frontal,
        "rotation_vector": rotation_vector.flatten().tolist(),
    }
    


def _classify_head_pose(yaw: float, pitch: float, roll: float) -> str:
    """
    Classifies the head pose into a human-readable orientation label.

    The classification is based on empirical thresholds calibrated for 
    standard webcam distances and perspectives.

    Threshold logic:
        - frontal  : yaw within [-15, 15] and pitch within [-15, 15]
        - left     : yaw < -15
        - right    : yaw >  15
        - up       : pitch > 15
        - down     : pitch < -15
        - tilted   : absolute roll > 20

    Args:
        yaw (float): Horizontal rotation angle in degrees.
        pitch (float): Vertical rotation angle in degrees.
        roll (float): Side-to-tilt rotation angle in degrees.

    Returns:
        str: Pose classification label. Possible values: "tilted", 
            "looking_left", "looking_right", "looking_up", 
            "looking_down", or "frontal".
            
    """
    YAW_THRESHOLD   = 15.0
    PITCH_THRESHOLD = 15.0
    ROLL_THRESHOLD  = 20.0

    # Priority check: Significant side tilt
    if abs(roll) > ROLL_THRESHOLD:
        
        return "tilted"
    
    # Horizontal orientation check
    if yaw < -YAW_THRESHOLD:
        
        return "looking_left"
    
    if yaw > YAW_THRESHOLD:
        
        return "looking_right"
    
    # Vertical orientation check
    if pitch > PITCH_THRESHOLD:
        
        return "looking_up"
    
    if pitch < -PITCH_THRESHOLD:
        
        return "looking_down"
    
    # Default state if within all thresholds
    return "frontal"


# Analyze Face Geometry
def analyze_face_geometry(
    landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    consecutive_low_ear_frames: int = 0
) -> dict:
    """
    Executes  facial geometric analysis in a single call.
    
    This is a high-level interface designed to be called by the streaming 
    module on every frame. It aggregates EAR (eyes), MAR (mouth), and 
    head pose data into a single structured response.

    Args:
        landmarks (np.ndarray): A (5, 2) array of SCRFD keypoints.
        image_width (int): Width of the original BGR frame in pixels.
        image_height (int): Height of the original BGR frame in pixels.
        consecutive_low_ear_frames (int): Counter for consecutive frames 
            with low EAR (managed by the streaming loop). Defaults to 0.

    Returns:
        dict: { : }
            - ear (dict): Eye state data and drowsiness flags.
            - mar (dict): Mouth aspect ratio and yawning status.
            - head_pose (dict): Orientation angles and pose classification.
            All data is formatted for JSON serialization and WebSocket transmission.
    """
    try:
        
        # Perform individual geometric calculations
        ear      = compute_ear_from_landmarks(landmarks)
        ear_data = classify_eye_state(ear, consecutive_low_ear_frames)

        mar      = compute_mar_from_landmarks(landmarks)
        mar_data = {
            "mar"         : mar,
            "is_yawning"  : mar > 0.60, # Threshold for yawn detection
        }
        
        # Estimate head orientation using PnP
        head_pose = estimate_head_pose(landmarks, image_width, image_height)

        return {
            "ear"      : ear_data,
            "mar"      : mar_data,
            "head_pose": head_pose,
        }

    except Exception as e:
        # Graceful degradation: log the error and return a neutral state
        logger.error(
            f"Error in analyze_face_geometry: {e}",
            exc_info=True
        )
        
        return {
            
            "ear"      : {"ear": 0.0, "eye_state": "unknown", "is_blinking": False, "is_drowsy": False},
            "mar"      : {"mar": 0.0, "is_yawning": False},
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "pose_label": "unknown", "is_frontal": False},
            
        }
