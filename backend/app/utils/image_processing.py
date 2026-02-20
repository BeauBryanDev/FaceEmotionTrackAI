import cv2
import numpy as np
import base64
from typing import Tuple, Optional

def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Decodes a Base64 string into an OpenCV image array.
    
    Handles standard web-encoded strings that may include data URI schemes 
    (e.g., 'data:image/jpeg;base64,...').
    
    Args:
        base64_string (str): The raw Base64 string from the WebSocket client.
        
    Returns:
        Optional[np.ndarray]: The decoded image in BGR format, or None if decoding fails.
    """
    try:
        # Strip the metadata header if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
            
        # Decode the Base64 string into raw bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to a 1D NumPy array of unsigned 8-bit integers
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode the 1D array into a 3D OpenCV image matrix (H, W, Channels)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        print(f"Error decoding Base64 image: {str(e)}")
        return None

def convert_and_resize(
    image: np.ndarray, 
    target_size: Tuple[int, int], 
    to_rgb: bool = True
) -> np.ndarray:
    """
    Resizes the image and optionally converts the color space from BGR to RGB.
    
    Args:
        image (np.ndarray): The input OpenCV image (BGR).
        target_size (Tuple[int, int]): The desired (width, height) output size.
        to_rgb (bool): If True, converts the image to RGB color space.
        
    Returns:
        np.ndarray: The processed image matrix.
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    if to_rgb:
        return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
    return resized_image

def prepare_tensor_for_onnx(
    image: np.ndarray, 
    mean: float = 0.5, 
    std: float = 0.5
) -> np.ndarray:
    """
    Normalizes the image and transposes dimensions for ONNX Runtime inference.
    
    Standard ONNX models require inputs in NCHW format (Batch, Channels, Height, Width)
    and normalized pixel values.
    
    Args:
        image (np.ndarray): The RGB image array of shape (Height, Width, Channels).
        mean (float): The mean value for normalization.
        std (float): The standard deviation for normalization.
        
    Returns:
        np.ndarray: A 4D tensor of shape (1, Channels, Height, Width) ready for inference.
    """
    # Scale pixel values from [0, 255] to [0.0, 1.0]
    normalized_img = image.astype(np.float32) / 255.0
    
    # Apply mean and standard deviation normalization
    normalized_img = (normalized_img - mean) / std
    
    # Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    chw_image = np.transpose(normalized_img, (2, 0, 1))
    
    # Add the batch dimension (N) to create NCHW shape
    tensor = np.expand_dims(chw_image, axis=0)
    
    return tensor