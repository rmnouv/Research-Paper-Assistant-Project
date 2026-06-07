from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Research Paper Assistant"
    openai_api_key: str | None = None
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_collection: str = "research_papers"
    chroma_persist_dir: str = "data/chroma"
    upload_dir: str = "data/uploads"
    chunk_size: int = 900
    chunk_overlap: int = 160
    retrieval_k: int = 12
    rerank_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
