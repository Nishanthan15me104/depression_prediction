"""
Security module.
Handles Authentication (AuthN) and Authorization (AuthZ).
"""
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.api.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the incoming API key."""
    if api_key_header != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API credentials"
        )
    return api_key_header