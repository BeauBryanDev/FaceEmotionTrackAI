from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field

class Settings(BaseSettings):
    PROJECT_NAME: str = "FaceEmotionTrackAI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # Database components (Pydantic reads these from .env)
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    POSTGRES_DB: str

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        # Build the URL: postgresql://user:pass@host:port/db
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"
    
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MODELS_PATH: str = "/app/ml_weights"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" 
    )
    
# Instantiate the settings object to be imported across the application
settings = Settings()