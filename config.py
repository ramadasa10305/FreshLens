"""Load and validate FreshLens settings from the project-root .env file.

Values are read once at import and exposed as module constants. API keys are
never printed, logged, or included in error messages.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _PROJECT_ROOT / ".env"

_REQUIRED_VARIABLES = (
    "BIFROST_BASE_URL",
    "BIFROST_API_KEY",
    "BIFROST_MODEL",
    "TAVILY_API_KEY",
)

# Values shipped in .env.example use a "replace_with..." prefix.
_PLACEHOLDER_PREFIX = "replace_with"
_PLACEHOLDER_VALUES = frozenset(
    {
        "changeme",
        "placeholder",
        "your_api_key",
        "your-api-key",
        "xxx",
        "todo",
        "none",
        "null",
        "undefined",
    }
)


def _is_placeholder(value: str) -> bool:
    """Return True when a value is blank or still an example placeholder."""
    normalized = value.strip().lower()
    if not normalized or normalized in _PLACEHOLDER_VALUES:
        return True
    if normalized.startswith(_PLACEHOLDER_PREFIX):
        return True
    return normalized.startswith("<") and normalized.endswith(">")


def _read_required(name: str) -> str:
    """Return a required variable, or raise without echoing its value."""
    raw = os.getenv(name)
    if raw is None or _is_placeholder(raw):
        raise RuntimeError(
            f"Required environment variable {name} is missing or still set "
            f"to a placeholder. Set a real value in {_ENV_FILE.name} at the "
            "project root."
        )
    return raw.strip()


def _is_http_url(value: str) -> bool:
    """Return True for an absolute HTTP or HTTPS URL with a valid host and port."""
    import ipaddress

    try:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"}:
            return False
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False

    if not hostname or (port is not None and not 1 <= port <= 65535):
        return False

    # urlparse treats an empty port, as in "http://host:", as omitted.
    hostinfo = parsed.netloc.rpartition("@")[2]
    if hostinfo.startswith("["):
        bracket_end = hostinfo.find("]")
        if bracket_end == -1 or hostinfo[bracket_end + 1 :] == ":":
            return False
    elif hostinfo.endswith(":"):
        return False

    try:
        if ":" in hostname:
            ipaddress.IPv6Address(hostname)
            return True
        labels = hostname.split(".")
        if len(labels) == 4 and all(label.isascii() and label.isdigit() for label in labels):
            ipaddress.IPv4Address(hostname)
            return True
    except ValueError:
        return False

    if len(hostname) > 253:
        return False
    for label in labels:
        if (
            not 1 <= len(label) <= 63
            or label.startswith("-")
            or label.endswith("-")
            or not all(char.isascii() and (char.isalnum() or char == "-") for char in label)
        ):
            return False
    return True


class _Settings(BaseModel):
    """Validated configuration. Secret fields are omitted from repr."""

    model_config = ConfigDict(frozen=True)

    bifrost_base_url: str
    bifrost_api_key: str = Field(repr=False)
    bifrost_model: str
    tavily_api_key: str = Field(repr=False)

    @field_validator("bifrost_base_url")
    @classmethod
    def validate_bifrost_base_url(cls, value: str) -> str:
        if not _is_http_url(value):
            raise ValueError("BIFROST_BASE_URL must be an HTTP or HTTPS URL.")
        return value


def _load_settings() -> _Settings:
    """Load .env from the project root and validate the required variables."""
    load_dotenv(_ENV_FILE)
    try:
        return _Settings(
            bifrost_base_url=_read_required("BIFROST_BASE_URL"),
            bifrost_api_key=_read_required("BIFROST_API_KEY"),
            bifrost_model=_read_required("BIFROST_MODEL"),
            tavily_api_key=_read_required("TAVILY_API_KEY"),
        )
    except ValidationError:
        # ValidationError text can repeat the submitted URL. Replace it with
        # a fixed message and drop the original exception.
        raise RuntimeError(
            "BIFROST_BASE_URL must be a valid HTTP or HTTPS URL."
        ) from None


_settings = _load_settings()

BIFROST_BASE_URL = _settings.bifrost_base_url
BIFROST_API_KEY = _settings.bifrost_api_key
BIFROST_MODEL = _settings.bifrost_model
TAVILY_API_KEY = _settings.tavily_api_key

__all__ = [
    "BIFROST_API_KEY",
    "BIFROST_BASE_URL",
    "BIFROST_MODEL",
    "TAVILY_API_KEY",
]
