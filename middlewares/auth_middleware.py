"""
Authentication middleware using hashed API key comparison.

Instead of comparing API keys in plaintext, the configured API_KEY in .env
is hashed at startup and all incoming tokens are hashed before comparison.
This prevents timing attacks and avoids storing secrets in memory as plaintext.
"""
import hashlib
import hmac
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from config.settings import get_settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _hash_key(key: str) -> str:
    """Produces a SHA-256 digest of the given key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def verify_api_key(
    raw_header: str = Security(api_key_header),
) -> str:
    """
    FastAPI dependency that validates the Authorization header.

    Expected format: ``Bearer <token>``

    The token is hashed and compared against the hashed version of
    ``settings.api_key`` using constant-time comparison to prevent
    timing side-channel attacks.
    """
    settings = get_settings()

    if not raw_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    parts = raw_header.split(" ", maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    token = parts[1]

    # Constant-time comparison of hashed values
    if not hmac.compare_digest(_hash_key(token), _hash_key(settings.api_key)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return token
