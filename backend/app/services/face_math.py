from statistics import variance
import numpy as np
from typing import Tuple, List, Dict, Any

def compute_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two high-dimensional vectors.
    
    ArcFace (w600k_mbf.onnx) typically outputs L2-normalized embeddings, 
    meaning the dot product alone would theoretically suffice. However, 
    calculating the full cosine similarity ensures stability even if raw 
    logits or unnormalized vectors are passed.
    
    Args:
        vector1 (np.ndarray): The first embedding vector (e.g., from the database).
        vector2 (np.ndarray): The second embedding vector (e.g., from the live webcam).
        
    Returns:
        float: A similarity score between -1.0 and 1.0. Higher means more similar.
    """
    # Ensure 1D arrays
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Prevent division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return float(dot_product / (norm_vec1 * norm_vec2))

def verify_biometric_match(
    stored_embedding: np.ndarray, 
    live_embedding: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Determines if two face embeddings belong to the same person.
    
    Args:
        stored_embedding (np.ndarray): The master embedding from user registration.
        live_embedding (np.ndarray): The real-time embedding from the stream.
        threshold (float): The boundary for acceptance. For ArcFace cosine similarity, 
                           0.45 to 0.50 is the industry standard threshold.
                           
    Returns:
        Tuple[bool, float]: A boolean indicating a match, and the exact similarity score.
    """
    similarity = compute_cosine_similarity(stored_embedding, live_embedding)
    is_match = similarity >= threshold
    
    return is_match, similarity

def apply_pca_reduction(embeddings: np.ndarray, n_components: int = 128) -> np.ndarray:
    """
    Reduces the dimensionality of a batch of embeddings using PCA via SVD.
    
    This is an advanced technique useful for optimizing vector storage or 
    preparing 512D data for 2D/3D visualization (e.g., t-SNE clustering).
    Implemented strictly with NumPy to demonstrate foundational linear algebra skills.
    
    Args:
        embeddings (np.ndarray): A 2D array of shape (n_samples, n_features).
        n_components (int): The target number of dimensions (e.g., reducing 512 to 128).
        
    Returns:
        
        Dict[str, Any]:
            {
                "reduced_embeddings": np.ndarray,
                "principal_components": np.ndarray,
                "explained_variance": np.ndarray,
                "explained_variance_ratio": np.ndarray,
                "cumulative_variance": np.ndarray,
                "mean_vector": np.ndarray
            }
        
    """
    
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2D array of shape (n_samples, n_features). "
            f"Got shape: {embeddings.shape}"
        )
    
    if embeddings.shape[1] <= n_components:
        return embeddings
    
    
    n_samples, n_features = embeddings.shape

    if n_features <= n_components:
        return {
            "reduced_embeddings": embeddings,
            "principal_components": None,
            "explained_variance": None,
            "explained_variance_ratio": None,
            "cumulative_variance": None,
            "mean_vector": None
        }

    # 1. Mean center data
    mean_vector = np.mean(embeddings, axis=0)
    centered_data = embeddings - mean_vector

    # limit components safely
    max_components = min(n_samples, n_features)
    n_components = min(n_components, max_components)

    # 2. SVD
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

    # 3. principal components
    principal_components = Vt[:n_components]

    # 4. projection
    reduced_data = np.dot(centered_data, principal_components.T)

    # 5. variance computation
    explained_variance = (S ** 2) / (n_samples - 1)

    # keep only selected components
    explained_variance = explained_variance[:n_components]

    total_variance = np.sum((S ** 2) / (n_samples - 1))
    # X = U * S * V^T
    explained_variance_ratio = explained_variance / total_variance
    # We use full_matrices=False for computational efficiency on large datasets.
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return {
        "reduced_embeddings": reduced_data,
        "principal_components": principal_components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "mean_vector": mean_vector
    }

    
def apply_pca_reduction_batch(
    embeddings: List[np.ndarray],
    n_components: int = 128
) -> List[Dict[str, Any]]:
    """
    Applies PCA reduction to multiple embedding datasets.

    Each element must be a matrix (n_samples, n_features).

    Args:
        embeddings: list of embedding matrices
        n_components: target PCA dimensions

    Returns:
        List of PCA result dictionaries
    """

    results = []

    for matrix in embeddings:

        if matrix.ndim != 2:
            raise ValueError(
                f"Each embedding batch must be (n_samples, n_features). Got {matrix.shape}"
            )

        pca_result = apply_pca_reduction(matrix, n_components)

        results.append(pca_result)

    return results
