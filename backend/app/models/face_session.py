from sqlalchemy import Column, Integer, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.core.database import Base


class FaceSessionEmbedding(Base):
    """
    Stores 512-dimensional ArcFace embeddings captured during live
    WebSocket sessions. These embeddings feed the PCA analytics endpoint
    to generate the 3D latent space scatter plot.

    A new record is inserted every N frames (configurable via settings)
    to avoid flooding the database with redundant vectors.
    """

    __tablename__ = "face_session_embeddings"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to the authenticated user who owns this embedding.
    # CASCADE ensures cleanup when a user account is deleted (GDPR compliance).
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # The 512-dimensional L2-normalized ArcFace embedding vector.
    # Stored as pgvector Vector type for efficient cosine similarity queries.
    embedding = Column(Vector(512), nullable=False)

    # Optional session identifier to group embeddings from the same connection.
    # Format: "{user_id}_{timestamp_unix}" generated in stream.py.
    session_id = Column(String(64), nullable=True, index=True)

    # Automatic timestamp for time-series analysis and data retention policies.
    captured_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )

    # Relationship back to User model.
    user = relationship("User", back_populates="session_embeddings")

    def __repr__(self) -> str:
        return (
            f"<FaceSessionEmbedding("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"session_id={self.session_id})>"
        )
