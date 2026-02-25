from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.config import settings
from app.core.session import get_db
from app.models.users import User
from app.core.security import ALGORITHM

# This schema is used internally to validate the token payload structure.

class TokenPayload(BaseModel):
    sub: str = None

# OAuth2 setup pointing to the login endpoint.
# This integration allows Swagger UI to automatically append the Bearer token 
# to protected endpoints during manual API testing.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)

def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    """
    Dependency function to extract and validate the JWT token from the incoming request.
    
    Args:
        db (Session): The active database session injected by FastAPI.
        token (str): The Bearer token extracted from the Authorization header.
        
    Returns:
        User: The SQLAlchemy User model instance corresponding to the token subject.
        
    Raises:
        HTTPException: If the token is invalid, expired, or the user no longer exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token using the secret key and the specified algorithm.
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        # Extract the subject (user ID) from the payload.
        token_data = TokenPayload(sub=str(payload.get("sub")))
        if token_data.sub is None:
            raise credentials_exception
            
    except JWTError:
        # Catches expired signatures or cryptographically invalid tokens.
        raise credentials_exception
        
    # Query the database to ensure the user still exists.
    user = db.query(User).filter(User.id == int(token_data.sub)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )
        
    return user

def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency function to ensure the authenticated user account is currently active.
    
    Args:
        current_user (User): The user object injected by get_current_user.
        
    Returns:
        User: The active user object.
        
    Raises:
        HTTPException: If the user account has been disabled or suspended.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user account"
        )
    return current_user


def get_current_active_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Dependency function to ensure the authenticated user account is currently active.
    
    Args:
        current_user (User): The user object injected by get_current_user.
        
    Returns:
        User: The active user object.
        
    Raises:
        HTTPException: If the user account has been disabled or suspended.
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
    )
        
    return current_user

async def get_user_from_token(token: str, db: Session) -> User:

    """

        Validates the JWT token passed via the WebSocket connection URL.

        Args:

        token (str): The JWT string.

        db (Session): The active database session.

        Returns:

        User: The authenticated user object, or None if validation fails.

    """

    try:

        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])

        user_id = payload.get("sub")

        if user_id is None:

            return None


        user = db.query(User).filter(User.id == int(user_id)).first()

        return user

    except (JWTError, ValueError, TypeError):
        
        return None
