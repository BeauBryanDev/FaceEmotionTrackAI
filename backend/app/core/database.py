import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# Configure basic logging for database events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the SQLAlchemy engine
# pool_pre_ping=True ensures that connections are validated before being used,
# preventing errors if the PostgreSQL container restarts or a connection drops.
# pool_size and max_overflow help manage concurrent connections efficiently.
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=15
)

# SessionLocal is a factory for new Session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all declarative SQLAlchemy models
Base = declarative_base()

def init_db():
    """
    Initializes database prerequisites.
    Crucially, it ensures the pgvector extension is enabled at the database level.
    This must run before any models attempt to use the Vector column type.
    """
    try:
        
        with engine.connect() as connection:
            
            # Execute raw SQL to enable the vector extension
            
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            connection.commit()
            
        logger.info("Successfully verified pgvector extension in PostgreSQL.")
        
    except Exception as e:
        
        logger.error(f"Failed to initialize database extensions: {e}")
        
        raise e

def get_db():
    """
    FastAPI dependency function to provide a database session per request.
    Yields a session and safely closes it after the HTTP request completes,
    preventing memory leaks and connection exhaustion.
    """
    db = SessionLocal()
    try:
        
        yield db
        
    finally:
        
        db.close()
