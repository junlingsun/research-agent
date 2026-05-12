"""Shared pytest fixtures and configuration."""

import os

# Set test environment variables before any app imports
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("GROQ_API_KEY", "gsk-test-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key-32-chars-long-xx")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("LLM_MODEL", "llama-3.3-70b-versatile")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-key")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/research_agent_test",
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("ENVIRONMENT", "test")
