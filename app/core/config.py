from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # App
    app_name: str = "Research Agent API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "production"

    # LLM provider
    llm_provider: str = "groq"
    llm_model: str = "Llama-3.3-70B-Versatile"

    # API Keys
    openai_api_key: str
    groq_api_key: str
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "research-agent"

    # Search
    tavily_api_key: str = ""

    # Qdrant (vector DB for memory)
    qdrant_url: str = "http://qdrant:6333"
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model

    # Security
    secret_key: str
    api_key_header: str = "X-API-Key"

    # Database
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/research_agent"
    )

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Agent settings
    agent_max_iterations: int = 10
    agent_max_search_results: int = 5
    agent_request_timeout: int = 30

    # Rate limiting
    rate_limit_per_minute: int = 20

    # Cache TTL (seconds)
    cache_ttl: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
