"""
API key authentication – validates requests against a list of allowed keys
loaded from the ``API_KEYS`` environment variable (comma-separated).

When ``API_KEYS`` is empty or unset the module operates in **dev mode**:
all requests are allowed so local development works without extra setup.
"""

from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()


def _load_api_keys() -> List[str]:
    """Return the list of valid API keys from the environment."""
    raw = os.getenv("API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys


# Module-level cache so we don't re-parse on every request.
_VALID_KEYS: List[str] = _load_api_keys()
_DEV_MODE: bool = len(_VALID_KEYS) == 0

if _DEV_MODE:
    import logging
    logging.getLogger("api.auth").warning(
        "API_KEYS not set — running in dev mode (no auth required)"
    )


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    """
    FastAPI dependency that enforces API-key authentication.

    In dev mode (no keys configured) every request is allowed.
    Otherwise the caller must send a valid key in the ``x-api-key`` header.

    Returns
    -------
    str | None
        The validated API key, or ``None`` in dev mode.

    Raises
    ------
    HTTPException 401
        If the key is missing or not in the allow-list.
    """
    if _DEV_MODE:
        return None

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if x_api_key not in _VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key
