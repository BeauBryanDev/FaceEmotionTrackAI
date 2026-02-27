import numpy as np
from typing import Optional
from sqlalchemy.orm import Session

from app.models.users import User
from app.models.face_session import FaceSessionEmbedding
from app.core.logging import get_logger


logger = get_logger(__name__)

MIN_SAMPLES_REQUIRED = 3

# Number of output dimensions for the 3D scatter plot.
N_COMPONENTS = 3

# Number of principal components to retain for the 3D scatter plot.
N_COMPONENTS_RETAINED = 2

def compute_pca(
    embeddings: np.ndarray,
    n_components: int = N_COMPONENTS
) -> dict:
    
    """
    Computes PCA on a batch of L2-normalized face embeddings using SVD.

    Args:
        embeddings:   2D array of shape (n_samples, 512). Each row is a
                      L2-normalized ArcFace embedding vector.
        n_components: Number of principal components to retain (default: 3).

    Returns:
        dict containing:
            projected:          ndarray (n_samples, n_components) - reduced coords.
            explained_variance: list of variance explained per component (0-1).
            total_variance:     float - cumulative variance explained by all components.
            mean_vector:        ndarray (512,) - mean used for centering (for inference).
            principal_components: ndarray (n_components, 512) - top eigenvectors.
    """
    
    n_samples, n_features = embeddings.shape
    logger.debug(f"Computing PCA on {n_samples} samples with {n_features} features.")
    
    if n_samples < MIN_SAMPLES_REQUIRED:
        
        raise ValueError(
            f"At least {MIN_SAMPLES_REQUIRED} samples are required for PCA. Got {n_samples}."
        )
        
        
    if   n_components > min(n_samples, n_features):
        
        n_components = min(n_samples, n_features)
        
        logger.warning(
            f"n_components reduced to {n_components} "
            f"(min of n_samples={n_samples}, n_features={n_features})."
        )

    # Mean centering - removes the DC component of the distribution.
    # Center the data
    mean_vector = np.mean(embeddings, axis=0)
    centered_embeddings = embeddings - mean_vector
    
    # Thin SVD decomposition
    # X = U * S * Vt
    # U:  (n_samples, n_samples)  - left singular vectors
    # S:  (min(n,p),)             - singular values (sqrt of eigenvalues)
    # Vt: (n_features, n_features) - right singular vectors (principal components)
    
    U, S, Vt = np.linalg.svd(centered_embeddings, full_matrices=False)
    
    # Compute explained variance ratio.
    # Eigenvalues of the covariance matrix = S^2 / (n_samples - 1)
    eigenvalues          = (S ** 2) / (n_samples - 1)
    total_eigenvalue_sum = float(np.sum(eigenvalues))
    
    explained_variance = []
    
    for i in range(n_components):
        
        ratio = float(eigenvalues[i] / total_eigenvalue_sum) if total_eigenvalue_sum > 0 else 0.0
        explained_variance.append(round(ratio, 4))

    total_variance = round(sum(explained_variance), 4)
    
    # Project data onto the top n_components principal components.
    
    principal_components = Vt[:n_components]
    
    # Projection: (n_samples, 512) @ (512, n_components) = (n_samples, n_components)
    
    projected = centered_embeddings @ principal_components.T
    
    logger.info(
        f"PCA computed. samples={n_samples} components={n_components} "
        f"variance_explained={total_variance:.4f}"
    )

    return {
        "projected"           : projected,
        "explained_variance"  : explained_variance,
        "total_variance"      : total_variance,
        "mean_vector"         : mean_vector,
        "principal_components": principal_components,
    }


# DATABASE QUERY HELPERS

def fetch_registered_embeddings(db: Session) -> list[dict]:
    
    """
    Fetches the master face embedding for every registered user.
    Returns only users who have completed biometric enrollment.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        List of dicts with keys: user_id, full_name, embedding.
    """
    
    users = (
        db.query(User)
        .filter(User.face_embedding.isnot(None))
        .filter(User.is_active == True)
        .all()
    )
    
    result = []
    
    for user in users:
        
        embedding = np.array(user.face_embedding, dtype=np.float32)
        
        result.append({
            "user_id"  : user.id,
            "full_name": user.full_name,
            "source"   : "registered",
            "embedding": embedding,
        })
        
        
    logger.info(f"Fetched {len(result)} registered embeddings from users table.")
    
    return result


def fetch_session_embeddings(
    db: Session,
    user_id: Optional[int] = None,
    limit: int = 200
) -> list[dict]:
    
    """
    Fetches session embeddings from face_session_embeddings table.
    Optionally filtered by user_id.

    Args:
        db:      Active SQLAlchemy session.
        user_id: If provided, returns only embeddings for that user.
        limit:   Maximum number of session embeddings to return.
                 Capped at 200 to avoid memory pressure on t3.small.

    Returns:
        List of dicts with keys: user_id, session_id, captured_at, embedding.
    """
    
    query = db.query(FaceSessionEmbedding)

    if user_id is not None:
        query = query.filter(FaceSessionEmbedding.user_id == user_id)

    records = (
        query
        .order_by(FaceSessionEmbedding.captured_at.desc())
        .limit(limit)
        .all()
    )
    
    result = []
    
    for record in records:
        
        embedding = np.array(record.embedding, dtype=np.float32)
        
        result.append({
            "user_id"   : record.user_id,
            "session_id": record.session_id,
            "source"    : "session",
            "captured_at": record.captured_at.isoformat(),
            "embedding" : embedding,
        })

    logger.info(
        f"Fetched {len(result)} session embeddings. "
        f"filter_user_id={user_id}"
    )
    
    return result

# MAIN ANALYTICS FUNCTION
def build_pca_payload(
    db: Session,
    requesting_user_id: int,
    include_session_embeddings: bool = True,
    session_limit: int = 200
) -> dict:
    """
    Builds the complete PCA analytics payload for the REST endpoint.
    
    Args:
        db:                         Active SQLAlchemy session.
        requesting_user_id:         ID of the authenticated user making the request.
        include_session_embeddings: Whether to include session embeddings.
        session_limit:              Max session embeddings to include.

    Returns:
        dict ready to serialize as JSON response with:
            points:              list of 3D coordinates with metadata.
            explained_variance:  variance explained per component.
            total_variance:      cumulative variance explained.
            total_points:        total number of embeddings used.
            registered_count:    number of registered user embeddings.
            session_count:       number of session embeddings.
            
    """
    # Collect all embeddings with their metadata
    registered = fetch_registered_embeddings(db)
    
    session_data = []
    
    if include_session_embeddings:
        
        session_data = fetch_session_embeddings(
            
            db,
            limit=session_limit
            
        )

    all_records = registered + session_data
    total_points = len(all_records)
    
    if total_points < MIN_SAMPLES_REQUIRED:
        
        logger.warning(
            f"Not enough embeddings for PCA. "
            f"Found {total_points}, need {MIN_SAMPLES_REQUIRED}."
        )
        
        return {
            "points"            : [],
            "explained_variance": [],
            "total_variance"    : 0.0,
            "total_points"      : total_points,
            "registered_count"  : len(registered),
            "session_count"     : len(session_data),
            "error"             : (
                f"Insufficient data for PCA. "
                f"Need at least {MIN_SAMPLES_REQUIRED} embeddings. "
                f"Found {total_points}."
            )
        }
        
    # Stack all embeddings into a single matrix (n_samples, 512)
    embedding_matrix = np.stack(
        
        [record["embedding"] for record in all_records], # List of ndarrays (n_samples, 512)
        axis=0
        
    ).astype(np.float32)
    
    pca_result = compute_pca(embedding_matrix, n_components=N_COMPONENTS)
    projected  = pca_result["projected"]
    
    # Build the response points list with metadata for frontend rendering
    points = []
    
    for i, record in enumerate(all_records):
        
        coords = projected[i].tolist()

        point = {
            "x"      : round(float(coords[0]), 6),
            "y"      : round(float(coords[1]), 6),
            "z"      : round(float(coords[2]), 6),
            "user_id": record["user_id"],
            "source" : record["source"],
        }
        
    
        # Include full_name only for registered embeddings
        if record["source"] == "registered":
            point["label"] = record.get("full_name", f"User {record['user_id']}")
        else:
            point["label"]      = f"Session {record.get('session_id', 'unknown')}"
            point["captured_at"] = record.get("captured_at")

        # Flag the requesting user's own points for highlighting in the frontend with React
        point["is_current_user"] = (record["user_id"] == requesting_user_id)
        
        points.append(point)
        
    
    return {
        "points"            : points,
        "explained_variance": pca_result["explained_variance"],
        "total_variance"    : pca_result["total_variance"],
        "total_points"      : total_points,
        "registered_count"  : len(registered),
        "session_count"     : len(session_data),
        "n_components"      : N_COMPONENTS,
        "embedding_dims"    : 512,
    }
    
    
