from pydantic_settings import BaseSettings
from pydantic import model_validator
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    # LLM provider settings
    OPENAI_API_KEY: str | None = None
    HUGGINGFACEHUB_API_TOKEN: str | None = None
    OPENAI_BASE_URL: str = "https://router.huggingface.co/v1"
    OPENAI_MODEL: str = "Qwen/Qwen3.5-397B-A17B:novita"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    TAVILY_API_KEY: str | None = None

    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Database settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # Retrieval settings
    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # New cache settings with type annotations
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @model_validator(mode="after")
    def _fallback_openai_key_from_hf(self):
        # Allow OpenAI-compatible clients to run against HF router tokens.
        if not self.OPENAI_API_KEY and self.HUGGINGFACEHUB_API_TOKEN:
            self.OPENAI_API_KEY = self.HUGGINGFACEHUB_API_TOKEN
        return self

settings = Settings()
