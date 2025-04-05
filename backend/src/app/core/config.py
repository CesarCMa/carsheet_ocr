from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Carsheet OCR API"
    
    # Add more settings as needed
    # For example:
    # UPLOAD_DIR: str = "uploads"
    # MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB

settings = Settings() 