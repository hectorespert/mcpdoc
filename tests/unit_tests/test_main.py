"""Tests for mcpdoc.main module."""

import asyncio

import pytest

from mcpdoc.main import (
    _get_fetch_description,
    _get_oauth2_domain_match,
    _get_oauth2_headers,
    _is_http_or_https,
    extract_domain,
)


def test_extract_domain() -> None:
    """Test extract_domain function."""
    # Test with https URL
    assert extract_domain("https://example.com/page") == "https://example.com/"

    # Test with http URL
    assert extract_domain("http://test.org/docs/index.html") == "http://test.org/"

    # Test with URL that has port
    assert extract_domain("https://localhost:8080/api") == "https://localhost:8080/"

    # Check trailing slash
    assert extract_domain("https://localhost:8080") == "https://localhost:8080/"

    # Test with URL that has subdomain
    assert extract_domain("https://docs.python.org/3/") == "https://docs.python.org/"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://example.com", True),
        ("https://example.com", True),
        ("/path/to/file.txt", False),
        ("file:///path/to/file.txt", False),
        (
            "ftp://example.com",
            False,
        ),  # Not HTTP or HTTPS, even though it's not a local file
    ],
)
def test_is_http_or_https(url, expected):
    """Test _is_http_or_https function."""
    assert _is_http_or_https(url) is expected


@pytest.mark.parametrize(
    "has_local_sources,expected_substrings",
    [
        (True, ["local file path", "file://"]),
        (False, ["URL to fetch"]),
    ],
)
def test_get_fetch_description(has_local_sources, expected_substrings):
    """Test _get_fetch_description function."""
    description = _get_fetch_description(has_local_sources)

    # Common assertions for both cases
    assert "Fetch and parse documentation" in description
    assert "Returns:" in description

    # Specific assertions based on has_local_sources
    for substring in expected_substrings:
        if has_local_sources:
            assert substring in description
        else:
            # For the False case, we only check that "local file path"
            # and "file://" are NOT present
            if substring in ["local file path", "file://"]:
                assert substring not in description


class _DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status_code = status_code

    def raise_for_status(self):
        if self._status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


class _DummyClient:
    def __init__(self):
        self.post_calls = 0

    async def post(self, *args, **kwargs):
        self.post_calls += 1
        return _DummyResponse({"access_token": "token123", "expires_in": 3600})


def test_get_oauth2_domain_match_uses_longest_prefix() -> None:
    oauth2_by_domain = {
        "https://example.com/": {"token_url": "https://auth.example.com/token", "client_id": "id", "client_secret": "secret"},
        "https://example.com/docs/": {"token_url": "https://auth.example.com/token", "client_id": "id2", "client_secret": "secret2"},
    }
    domain = _get_oauth2_domain_match("https://example.com/docs/llms.txt", oauth2_by_domain)
    assert domain == "https://example.com/docs/"


def test_get_oauth2_headers_returns_empty_when_not_configured() -> None:
    headers = asyncio.run(
        _get_oauth2_headers(
            url="https://example.com/llms.txt",
            httpx_client=_DummyClient(),
            oauth2_by_domain={},
            oauth2_token_cache={},
        )
    )
    assert headers == {}


def test_get_oauth2_headers_fetches_and_caches_token() -> None:
    client = _DummyClient()
    oauth2_by_domain = {
        "https://example.com/": {
            "token_url": "https://auth.example.com/token",
            "client_id": "id",
            "client_secret": "secret",
            "scope": "read:docs",
        }
    }
    token_cache = {}

    headers_1 = asyncio.run(
        _get_oauth2_headers(
            url="https://example.com/llms.txt",
            httpx_client=client,
            oauth2_by_domain=oauth2_by_domain,
            oauth2_token_cache=token_cache,
        )
    )
    headers_2 = asyncio.run(
        _get_oauth2_headers(
            url="https://example.com/docs/page.md",
            httpx_client=client,
            oauth2_by_domain=oauth2_by_domain,
            oauth2_token_cache=token_cache,
        )
    )

    assert headers_1 == {"Authorization": "Bearer token123"}
    assert headers_2 == {"Authorization": "Bearer token123"}
    assert client.post_calls == 1
