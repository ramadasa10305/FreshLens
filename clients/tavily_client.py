"""Client for the Tavily Search API.

The API key comes from config.py. It is never printed, logged, or included
in error messages. This module only retrieves search evidence.
"""

from __future__ import annotations
import ipaddress
import re
import math
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from config import TAVILY_API_KEY

SEARCH_URL = "https://api.tavily.com/search"
_REQUEST_TIMEOUT_SECONDS = 30

_TOPICS = frozenset({"general", "news", "finance"})
_SEARCH_DEPTHS = frozenset({"basic", "advanced", "fast", "ultra-fast"})
_TIME_RANGES = frozenset({"day", "week", "month", "year", "d", "w", "m", "y"})
_MIN_RESULTS = 1
_MAX_RESULTS = 20


@dataclass(frozen=True)
class TavilySearchResult:
    """One normalized search hit used as evidence."""

    title: str
    url: str
    content: str
    published_date: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class TavilySearchResponse:
    """Search evidence and credit usage, when Tavily reports it."""

    query: str
    results: tuple[TavilySearchResult, ...]
    credits: int | None = None


class TavilyClient:
    """Retrieve web evidence from the Tavily Search API."""

    def search(
        self,
        query: str,
        *,
        topic: str = "general",
        search_depth: str = "basic",
        max_results: int = 5,
        time_range: str | None = None,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> TavilySearchResponse:
        """Search Tavily and return normalized title, URL, excerpt, date, and score."""
        cleaned_query = _validate_query(query)
        _validate_choice("topic", topic, _TOPICS)
        _validate_choice("search_depth", search_depth, _SEARCH_DEPTHS)
        _validate_max_results(max_results)
        if time_range is not None:
            _validate_choice("time_range", time_range, _TIME_RANGES)
        _validate_timeout(timeout)

        payload: dict[str, object] = {
            "query": cleaned_query,
            "topic": topic,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_published_date": True,
            "include_usage": True,
        }
        if time_range is not None:
            payload["time_range"] = time_range

        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                SEARCH_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.Timeout:
            raise _failure(
                "Tavily request timed out before a response was received."
            ) from None
        except requests.ConnectionError:
            raise _failure("Could not connect to the Tavily Search API.") from None
        except requests.RequestException:
            raise _failure(
                "Tavily request failed before a response was received."
            ) from None

        if response.status_code >= 400:
            raise _failure(f"Tavily returned HTTP {response.status_code}.") from None

        try:
            data = response.json()
        except ValueError:
            raise _failure("Tavily returned a response that is not valid JSON.") from None

        return _normalize_response(cleaned_query, data)


def _validate_query(query: object) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")
    return query.strip()


def _validate_choice(name: str, value: object, allowed: frozenset[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}.")


def _validate_max_results(max_results: object) -> None:
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not _MIN_RESULTS <= max_results <= _MAX_RESULTS
    ):
        raise ValueError(
            f"max_results must be an integer from {_MIN_RESULTS} to {_MAX_RESULTS}."
        )


def _validate_timeout(timeout: object) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number of seconds.")
    try:
        is_finite = math.isfinite(timeout)
    except OverflowError:
        raise ValueError("timeout must be a positive number of seconds.") from None
    if not is_finite or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds.")


def _failure(message: str) -> RuntimeError:
    """Build an error that cannot repeat the API key."""
    if TAVILY_API_KEY and TAVILY_API_KEY in message:
        message = "Tavily request failed."
    return RuntimeError(message)


def _normalize_response(query: str, data: object) -> TavilySearchResponse:
    if not isinstance(data, dict):
        raise _failure("Tavily response did not include a results list.")

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise _failure("Tavily response did not include a results list.")

    results = tuple(_normalize_result(item) for item in raw_results)
    return TavilySearchResponse(
        query=query,
        results=results,
        credits=_normalize_credits(data.get("usage", None)),
    )


def _normalize_result(item: object) -> TavilySearchResult:
    if not isinstance(item, dict):
        raise _failure("Tavily response included a search result with an unexpected shape.")

    title = item.get("title")
    url = item.get("url")
    content = item.get("content")
    if (
        not isinstance(title, str)
        or not isinstance(url, str)
        or not _is_absolute_http_url(url)
        or not isinstance(content, str)
    ):
        raise _failure("Tavily response included a search result with an unexpected shape.")

    return TavilySearchResult(
        title=title,
        url=url,
        content=content,
        published_date=_normalize_published_date(item.get("published_date", None)),
        score=_normalize_score(item.get("score", None)),
    )


def _normalize_published_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    raise _failure("Tavily response included a search result with an unexpected shape.")




def _is_absolute_http_url(url: str) -> bool:
    """Validate an absolute HTTP(S) URL and its hostname."""

    if not url or any(
        char.isspace()
        or ord(char) < 32
        or ord(char) == 127
        or unicodedata.category(char) == "Cc"
        for char in url
    ):
        return False

    if "\\" in url:
        return False

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port
    except (ValueError, UnicodeError):
        return False

    if parsed.scheme not in {"http", "https"} or not hostname:
        return False

    # Source URLs must not contain embedded credentials.
    if parsed.username is not None or parsed.password is not None:
        return False

    host_part = parsed.netloc.rsplit("@", 1)[-1]

    if host_part.endswith(":"):
        return False

    if port is not None and not 1 <= port <= 65535:
        return False

    # Accept valid IPv4/IPv6 addresses.
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Validate domain names, including internationalized domains.
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if ascii_hostname.endswith("."):
        ascii_hostname = ascii_hostname[:-1]

    if not ascii_hostname or len(ascii_hostname) > 253:
        return False

    labels = ascii_hostname.split(".")

    for label in labels:
        if not re.fullmatch(
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
            label,
        ):
            return False

    return True





def _normalize_score(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _failure("Tavily response included a search result with an unexpected shape.")
    try:
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise _failure("Tavily response included a search result with an unexpected shape.")
    except OverflowError:
        raise _failure("Tavily response included a search result with an unexpected shape.") from None
    return score


def _normalize_credits(usage: object) -> int | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise _failure("Tavily response included usage information with an unexpected shape.")
    credits = usage.get("credits", None)
    if credits is None:
        return None
    if isinstance(credits, bool) or not isinstance(credits, int):
        raise _failure("Tavily response included usage information with an unexpected shape.")
    return credits


__all__ = [
    "SEARCH_URL",
    "TavilyClient",
    "TavilySearchResponse",
    "TavilySearchResult",
]
