from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    COLLECTION_NAME: str = "pdf_docs"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MISTRAL_MODEL: str = "mistral-large-latest"
    TOP_K_RESULTS: int = 4
    CHROMA_DB_PATH: str = "./chroma_db"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
