import numpy as np
from typing import Tuple, List

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
    threshold: float = 0.45
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
        np.ndarray: The dimensionality-reduced embeddings.
    """
    if embeddings.shape[1] <= n_components:
        return embeddings

    # Step 1: Mean centering the data
    mean_vector = np.mean(embeddings, axis=0)
    centered_data = embeddings - mean_vector
    
    # Step 2: Singular Value Decomposition (SVD)
    # X = U * S * V^T
    # We use full_matrices=False for computational efficiency on large datasets.
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    
    # Step 3: Projection onto the principal components
    # We take the top 'n_components' from the transposed right singular vectors (Vt)
    principal_components = Vt.T[:, :n_components]
    reduced_data = np.dot(centered_data, principal_components)
    
    return reduced_data

def apply_pca_reduction_batch(
    embeddings: List[np.ndarray], n_components: int = 128
) -> List[np.ndarray]:
    """
    Reduces the dimensionality of a batch of embeddings using PCA via SVD.
    
    This is an advanced technique useful for optimizing vector storage or 
    preparing 512D data for 2D/3D visualization (e.g., t-SNE clustering).
    Implemented strictly with NumPy to demonstrate foundational linear algebra skills.
    
    Args:
        embeddings (List[np.ndarray]): A list of 2D arrays of shape (n_samples, n_features).
        n_components (int): The target number of dimensions (e.g., reducing 512 to 128).
        
    Returns:
        List[np.ndarray]: The dimensionality-reduced embeddings.
    """
    # Apply PCA reduction to each embedding in the batch
    reduced_embeddings = [
        apply_pca_reduction(embedding, n_components) for embedding in embeddings
    ]
    
    return reduced_embeddings   