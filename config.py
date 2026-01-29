from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_key: str = "dev-secret-key-12345"
    database_url: str = "sqlite:///./reviews.db"
    
    # AI Configuration
    cohere_api_key: Optional[str] = None
    use_local_model: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
