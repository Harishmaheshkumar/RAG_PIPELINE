from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # .env file-ah automatic-ah load panna
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Pipeline code-la use panna poren name GOOGLE_API_KEY. 
    # .env-la endha name irundhalum idhu handle pannum.
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY" 
    )
    
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")
    GEMINI_EMBEDDING_MODEL: str = Field(default="text-embedding-004")
    
    CHROMA_PERSIST_DIR: Path = Field(default=Path("./chroma_store"))
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    MLFLOW_EXPERIMENT: str = Field(default="rag_pipeline")
    
    # RAG Parameters
    MAX_RETRIEVAL_RESULTS: int = Field(default=10)
    RERANK_TOP_K: int = Field(default=5)
    RESPONSE_TEMPERATURE: float = Field(default=0.2)


settings = Settings()