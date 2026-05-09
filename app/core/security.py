from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from app.core.config import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)

# In production, store hashed keys in the database.
# This is a simple in-memory check for the scaffold.
VALID_API_KEYS: set[str] = {"dev-key-replace-in-production"}


async def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
