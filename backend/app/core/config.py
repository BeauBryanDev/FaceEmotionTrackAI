from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application configuration class.
    Reads variables from the environment or the .env file.
    Pydantic automatically validates the data types.
    """
    # Project Info
    PROJECT_NAME: str = "FaceEmotionTrackAI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Machine Learning configuration
    MODELS_PATH: str = "./ml_weights"

    # Pydantic V2 configuration to load the .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra variables in the environment that are not defined in this class
        extra="ignore" 
    )

# Instantiate the settings object to be imported across the application
settings = Settings()