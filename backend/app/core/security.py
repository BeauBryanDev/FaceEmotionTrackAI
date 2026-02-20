from datetime import datetime, timedelta, timezone
from typing import Any, Union
from jose import jwt
from passlib.context import CryptContext
from app.core.config import settings

# Initialize the password hashing context.
# IT explicitly sets the scheme to bcrypt. The "deprecated=auto" flag allows passlib 
# to handle older hashes gracefully if the algorithm is updated in the future.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration constants
# HS256 requires a single secret key for both signing and verification.
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain text password against a stored bcrypt hash.
    
    Args:
        plain_password (str): The raw password provided by the user during login.
        hashed_password (str): The bcrypt hash retrieved from the database.
        
    Returns:
        bool: True if the password matches the hash, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Generates a secure bcrypt hash for a given password.
    
    Args:
        password (str): The raw password provided during registration.
        
    Returns:
        str: The hashed password string ready to be stored in the database.
    """
    return pwd_context.hash(password)

def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    """
    Creates a JSON Web Token (JWT) using the HS256 algorithm.
    
    Args:
        subject (Union[str, Any]): The subject of the token, typically the user ID.
        expires_delta (timedelta, optional): Custom expiration time. If not provided, 
                                             defaults to ACCESS_TOKEN_EXPIRE_MINUTES.
                                             
    Returns:
        str: The encoded JWT string.
    """
    if expires_delta:
        
        expire = datetime.now(timezone.utc) + expires_delta
        
    else:
        
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # The payload strictly contains the standard "exp" (expiration time) 
    # and "sub" (subject identifier) claims.
    to_encode = {"exp": expire, "sub": str(subject)}
    
    # Sign the JWT using the secret key and the specified algorithm
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt