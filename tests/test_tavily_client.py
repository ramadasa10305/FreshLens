"""Mocked unit tests for the Tavily search client. No live API calls."""

import math
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clients import tavily_client
from clients.tavily_client import TavilyClient, TavilySearchResult

_FAKE_KEY = "test-key-should-not-leak"


def _response(*, status_code=200, payload=None, json_error=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = payload
    return response


def _result(**overrides):
    item = {
        "title": "t",
        "url": "https://example.com",
        "content": "c",
    }
    item.update(overrides)
    return {"results": [item]}


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setattr(tavily_client, "TAVILY_API_KEY", _FAKE_KEY)
    return _FAKE_KEY


def test_search_normalizes_results_and_requests_dates_and_usage(monkeypatch, fake_key):
    payload = {
        "query": "milk recall",
        "results": [
            {
                "title": "Recall notice",
                "url": "https://example.com/recall?id=1&view=full#section",
                "content": "A retailer recalled milk.",
                "score": 0.91,
                "published_date": "Tue, 11 Mar 2025 17:00:00 GMT",
                "raw_content": "ignored",
            },
            {
                "title": "No date",
                "url": "https://example.com/other",
                "content": "Excerpt only.",
                "score": 1,
            },
        ],
        "usage": {"credits": 1},
    }
    post = MagicMock(return_value=_response(payload=payload))
    monkeypatch.setattr(tavily_client.requests, "post", post)

    result = TavilyClient().search(
        "  milk recall  ",
        topic="news",
        search_depth="advanced",
        max_results=3,
        time_range="week",
        timeout=12,
    )

    assert result.query == "milk recall"
    assert result.credits == 1
    assert result.results == (
        TavilySearchResult(
            title="Recall notice",
            url="https://example.com/recall?id=1&view=full#section",
            content="A retailer recalled milk.",
            published_date="Tue, 11 Mar 2025 17:00:00 GMT",
            score=0.91,
        ),
        TavilySearchResult(
            title="No date",
            url="https://example.com/other",
            content="Excerpt only.",
            published_date=None,
            score=1.0,
        ),
    )

    kwargs = post.call_args.kwargs
    assert post.call_args.args == (tavily_client.SEARCH_URL,)
    assert kwargs["timeout"] == 12
    assert kwargs["headers"]["Authorization"] == f"Bearer {fake_key}"
    assert kwargs["json"] == {
        "query": "milk recall",
        "topic": "news",
        "search_depth": "advanced",
        "max_results": 3,
        "time_range": "week",
        "include_published_date": True,
        "include_usage": True,
    }
    assert "api_key" not in kwargs["json"]
    assert fake_key not in str(kwargs["json"])


def test_optional_date_score_and_usage_can_be_absent(monkeypatch, fake_key):
    payload = {
        "results": [
            {
                "title": "Untitled source",
                "url": "https://example.com/a",
                "content": "",
                "published_date": "   ",
                "score": None,
            }
        ]
    }
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(return_value=_response(payload=payload)),
    )

    result = TavilyClient().search("query")

    assert result.credits is None
    assert result.results[0].published_date is None
    assert result.results[0].score is None
    assert result.results[0].content == ""
    sent = tavily_client.requests.post.call_args.kwargs["json"]
    assert "time_range" not in sent
    assert sent["topic"] == "general"
    assert sent["search_depth"] == "basic"
    assert sent["max_results"] == 5
    assert fake_key not in str(sent)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"query": "   "}, "non-empty string"),
        ({"query": None}, "non-empty string"),
        ({"query": "ok", "topic": "sports"}, "topic must be one of"),
        ({"query": "ok", "search_depth": "deep"}, "search_depth must be one of"),
        ({"query": "ok", "max_results": 0}, "max_results"),
        ({"query": "ok", "max_results": 21}, "max_results"),
        ({"query": "ok", "max_results": True}, "max_results"),
        ({"query": "ok", "time_range": "decade"}, "time_range must be one of"),
        ({"query": "ok", "timeout": 0}, "timeout"),
        ({"query": "ok", "timeout": 0.0}, "timeout"),
        ({"query": "ok", "timeout": -1}, "timeout"),
        ({"query": "ok", "timeout": -0.5}, "timeout"),
        ({"query": "ok", "timeout": True}, "timeout"),
        ({"query": "ok", "timeout": False}, "timeout"),
        ({"query": "ok", "timeout": math.nan}, "timeout"),
        ({"query": "ok", "timeout": math.inf}, "timeout"),
        ({"query": "ok", "timeout": -math.inf}, "timeout"),
        ({"query": "ok", "timeout": 10**1000}, "timeout"),
        ({"query": "ok", "timeout": -(10**1000)}, "timeout"),
        ({"query": "ok", "timeout": "10"}, "timeout"),
        ({"query": "ok", "timeout": None}, "timeout"),
    ],
)
def test_invalid_input_is_rejected_without_a_request(monkeypatch, kwargs, match):
    post = MagicMock()
    monkeypatch.setattr(tavily_client.requests, "post", post)

    with pytest.raises(ValueError, match=match):
        TavilyClient().search(**kwargs)

    post.assert_not_called()


def test_http_error_hides_status_body_and_key(monkeypatch, fake_key):
    body = f'{{"detail":"invalid {fake_key}"}}'
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(return_value=_response(status_code=401, text=body)),
    )

    with pytest.raises(RuntimeError, match="Tavily returned HTTP 401") as exc_info:
        TavilyClient().search("query")

    message = str(exc_info.value)
    assert fake_key not in message
    assert "invalid" not in message
    displayed = _displayed_traceback(exc_info.value)
    assert fake_key not in displayed
    assert "invalid" not in displayed
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


def _displayed_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


@pytest.mark.parametrize(
    ("error_type", "detail", "match"),
    [
        (requests.Timeout, "timed-out-detail", "timed out"),
        (requests.ConnectionError, "connection-detail", "Could not connect"),
        (requests.RequestException, "request-detail", "failed before a response"),
    ],
)
def test_transport_errors_hide_original_details(monkeypatch, fake_key, error_type, detail, match):
    secret_detail = f"{detail} {fake_key}"
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": f"Bearer {fake_key}"}
    error_instance = error_type(secret_detail, request=mock_request)
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(side_effect=error_instance),
    )

    with pytest.raises(RuntimeError, match=match) as exc_info:
        TavilyClient().search("query")

    displayed = _displayed_traceback(exc_info.value)
    assert match in displayed
    assert fake_key not in displayed
    assert detail not in displayed
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"answer": "only"},
        {"results": "nope"},
        {"results": ["not-a-result"]},
        {"results": [{"title": "t", "content": "c"}]},
        {"results": [{"title": "t", "url": "   ", "content": "c"}]},
        {"results": [{"title": 1, "url": "https://example.com", "content": "c"}]},
        {"results": [{"title": "t", "url": "https://example.com", "content": 1}]},
        _result(score="high"),
        _result(score=[0.5]),
        _result(score=True),
        _result(score=False),
        _result(score=-0.01),
        _result(score=-1.0),
        _result(score=1.01),
        _result(score=2.0),
        _result(score=math.nan),
        _result(score=math.inf),
        _result(score=-math.inf),
        _result(score=10**1000),
        _result(score=-(10**1000)),
        {
            "results": [
                {
                    "title": "t",
                    "url": "https://example.com",
                    "content": "c",
                    "published_date": 20250101,
                }
            ]
        },
        {"results": [], "usage": ["credits"]},
        {"results": [], "usage": {"credits": 1.5}},
        {"results": [], "usage": {"credits": True}},
    ],
)
def test_malformed_responses(monkeypatch, payload):
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(return_value=_response(payload=payload)),
    )

    with pytest.raises(RuntimeError, match="Tavily response"):
        TavilyClient().search("query")


@pytest.mark.parametrize(
    "json_error_builder",
    [
        lambda fake_key: ValueError(f"bad-json-detail {fake_key}"),
        lambda fake_key: requests.exceptions.JSONDecodeError(
            f"bad-json-detail {fake_key}", "doc", 0
        ),
    ],
)
def test_invalid_json_hides_parser_details(monkeypatch, fake_key, json_error_builder):
    error = json_error_builder(fake_key)
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(
            return_value=_response(
                json_error=error,
                text=fake_key,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="not valid JSON") as exc_info:
        TavilyClient().search("query")

    displayed = _displayed_traceback(exc_info.value)
    assert "not valid JSON" in displayed
    assert fake_key not in displayed
    assert "bad-json-detail" not in displayed
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("url", "score"),
    [
        ("https://example.com/recall?id=1&view=full#section", 0.0),
        ("http://127.0.0.1:8080/other", 1.0),
        ("https://news.example.com/path/to/article", 0.5),
        ("http://[::1]:8080/path", 1),
        ("http://[::1]/path", None),
        ("https://example.com:443/secure", 0.8),
        ("http://example.com:80/standard", 0.2),
        ("https://example.com:1/min-port", 0.3),
        ("https://example.com:65535/max-port", 0.4),
        ("https://example.com/café", 0.9),
    ],
)
def test_valid_source_urls_are_preserved(monkeypatch, url, score):
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(return_value=_response(payload=_result(url=url, score=score))),
    )

    result = TavilyClient().search("query")

    assert result.results[0].url == url
    expected_score = float(score) if score is not None else None
    assert result.results[0].score == expected_score


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        " https://example.com/recall",
        "https://example.com/recall ",
        "https://example.com/a b",
        "https://example.com/a\tb",
        "https://example.com/a\nb",
        "https://example.com/a\rb",
        "https://example.com/a\x00b",
        "https://example.com/a\x1fb",
        "https://example.com/a\x7fb",
        "https://example.com/a\x85b",
        "https://user:secret@example.com/a",
        "https://exa%mple.com",
        "https://-example.com",
        "https://example-.com",
        "https://example..com",
        "/recall",
        "example.com/recall",
        "//example.com/recall",
        "../recall",
        "ftp://example.com/file",
        "javascript:alert(1)",
        "file:///path",
        "data:text/html,x",
        "http://",
        "http:///",
        "http:///missing-host",
        "http://:8080/path",
        "http://?query=1",
        "http://#section",
        "http://user:pass@/path",
        "http://example.com:0/path",
        "http://example.com:65536/path",
        "http://example.com:99999/path",
        "http://example.com:-1/path",
        "http://example.com:abc/path",
        "http://example.com:80:80/path",
        "http://example.com:/path",
        "http://[::1]:/path",
        "http://example.com\\path",
        "http:\\\\example.com/path",
        "http://[::1/path",
        "http://[invalid-ipv6]/path",
    ],
)
def test_invalid_source_urls_are_rejected(monkeypatch, url):
    monkeypatch.setattr(
        tavily_client.requests,
        "post",
        MagicMock(return_value=_response(payload=_result(url=url))),
    )

    with pytest.raises(RuntimeError, match="unexpected shape"):
        TavilyClient().search("query")
