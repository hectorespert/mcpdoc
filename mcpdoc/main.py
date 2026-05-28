"""MCP Llms-txt server for docs."""

import os
import re
import time
from urllib.parse import urlparse, urljoin

import httpx
from markdownify import markdownify
from mcp.server.fastmcp import FastMCP
from typing_extensions import NotRequired, TypedDict


class DocSource(TypedDict):
    """A source of documentation for a library or a package."""

    name: NotRequired[str]
    """Name of the documentation source (optional)."""

    llms_txt: str
    """URL to the llms.txt file or documentation source."""

    description: NotRequired[str]
    """Description of the documentation source (optional)."""

    oauth2: NotRequired["OAuth2Config"]
    """OAuth2 client-credentials configuration for protected docs."""


class OAuth2Config(TypedDict):
    """OAuth2 client credentials configuration."""

    token_url: str
    client_id: str
    client_secret: str
    scope: NotRequired[str]
    audience: NotRequired[str]
    grant_type: NotRequired[str]


class OAuth2TokenCacheEntry(TypedDict):
    """OAuth2 access token cache entry."""

    access_token: str
    expires_at: float


def extract_domain(url: str) -> str:
    """Extract domain from URL.

    Args:
        url: Full URL

    Returns:
        Domain with scheme and trailing slash (e.g., https://example.com/)
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def _is_http_or_https(url: str) -> bool:
    """Check if the URL is an HTTP or HTTPS URL."""
    return url.startswith(("http:", "https:"))


def _get_fetch_description(has_local_sources: bool) -> str:
    """Get fetch docs tool description."""
    description = [
        "Fetch and parse documentation from a given URL or local file.",
        "",
        "Use this tool after list_doc_sources to:",
        "1. First fetch the llms.txt file from a documentation source",
        "2. Analyze the URLs listed in the llms.txt file",
        "3. Then fetch specific documentation pages relevant to the user's question",
        "",
    ]

    if has_local_sources:
        description.extend(
            [
                "Args:",
                "    url: The URL or file path to fetch documentation from. Can be:",
                "        - URL from an allowed domain",
                "        - A local file path (absolute or relative)",
                "        - A file:// URL (e.g., file:///path/to/llms.txt)",
            ]
        )
    else:
        description.extend(
            [
                "Args:",
                "    url: The URL to fetch documentation from.",
            ]
        )

    description.extend(
        [
            "",
            "If a source is configured with OAuth2, requests automatically use a bearer token.",
            "",
            "Returns:",
            "    The fetched documentation content converted to markdown, or an error message",  # noqa: E501
            "    if the request fails or the URL is not from an allowed domain.",
        ]
    )

    return "\n".join(description)


def _normalize_path(path: str) -> str:
    """Accept paths in file:/// or relative format and map to absolute paths."""
    return (
        os.path.abspath(path[7:])
        if path.startswith("file://")
        else os.path.abspath(path)
    )


def _get_oauth2_domain_match(url: str, oauth2_by_domain: dict[str, OAuth2Config]) -> str | None:
    """Find the best matching configured domain for a URL."""
    matches = [domain for domain in oauth2_by_domain if url.startswith(domain)]
    if not matches:
        return None
    return max(matches, key=len)


async def _get_oauth2_headers(
    url: str,
    httpx_client: httpx.AsyncClient,
    oauth2_by_domain: dict[str, OAuth2Config],
    oauth2_token_cache: dict[str, OAuth2TokenCacheEntry],
) -> dict[str, str]:
    """Return OAuth2 Authorization header when source domain is configured."""
    domain = _get_oauth2_domain_match(url, oauth2_by_domain)
    if domain is None:
        return {}

    cached_token = oauth2_token_cache.get(domain)
    now = time.time()
    if cached_token and cached_token["expires_at"] > now:
        return {"Authorization": f"Bearer {cached_token['access_token']}"}

    config = oauth2_by_domain[domain]
    token_request_data = {
        "grant_type": config.get("grant_type", "client_credentials"),
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
    }

    if config.get("scope"):
        token_request_data["scope"] = config["scope"]
    if config.get("audience"):
        token_request_data["audience"] = config["audience"]

    token_response = await httpx_client.post(config["token_url"], data=token_request_data)
    token_response.raise_for_status()
    token_payload = token_response.json()
    access_token = token_payload.get("access_token")
    if not access_token:
        raise ValueError("OAuth2 token response missing access_token")

    expires_in = token_payload.get("expires_in")
    token_ttl = float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0
    oauth2_token_cache[domain] = {
        "access_token": access_token,
        "expires_at": now + max(token_ttl - 30.0, 1.0),
    }

    return {"Authorization": f"Bearer {access_token}"}


def _get_server_instructions(doc_sources: list[DocSource]) -> str:
    """Generate server instructions with available documentation source names."""
    # Extract source names from doc_sources
    source_names = []
    for entry in doc_sources:
        if "name" in entry:
            source_names.append(entry["name"])
        elif _is_http_or_https(entry["llms_txt"]):
            # Use domain name as fallback for HTTP sources
            domain = extract_domain(entry["llms_txt"])
            source_names.append(domain.rstrip("/").split("//")[-1])
        else:
            # Use filename as fallback for local sources
            source_names.append(os.path.basename(entry["llms_txt"]))

    instructions = [
        "Use the list_doc_sources tool to see available documentation sources.",
        "This tool will return a URL for each documentation source.",
    ]

    if source_names:
        if len(source_names) == 1:
            instructions.append(
                f"Documentation URLs are available from this tool "
                f"for {source_names[0]}."
            )
        else:
            names_str = ", ".join(source_names[:-1]) + f", and {source_names[-1]}"
            instructions.append(
                f"Documentation URLs are available from this tool for {names_str}."
            )

    instructions.extend(
        [
            "",
            "Once you have a source documentation URL, use the fetch_docs tool "
            "to get the documentation contents. ",
            "If the documentation contents contains a URL for additional documentation "
            "that is relevant to your task, you can use the fetch_docs tool to "
            "fetch documentation from that URL next.",
        ]
    )

    return "\n".join(instructions)


def create_server(
    doc_sources: list[DocSource],
    *,
    follow_redirects: bool = False,
    timeout: float = 10,
    settings: dict | None = None,
    allowed_domains: list[str] | None = None,
) -> FastMCP:
    """Create the server and generate documentation retrieval tools.

    Args:
        doc_sources: List of documentation sources to make available
        follow_redirects: Whether to follow HTTP redirects when fetching docs
        timeout: HTTP request timeout in seconds
        settings: Additional settings to pass to FastMCP
        allowed_domains: Additional domains to allow fetching from.
            Use ['*'] to allow all domains
            The domain hosting the llms.txt file is always appended to the list
            of allowed domains.

    Returns:
        A FastMCP server instance configured with documentation tools
    """
    settings = settings or {}
    server = FastMCP(
        name="llms-txt",
        instructions=_get_server_instructions(doc_sources),
        **settings,
    )
    httpx_client = httpx.AsyncClient(follow_redirects=follow_redirects, timeout=timeout)

    local_sources = []
    remote_sources = []

    for entry in doc_sources:
        url = entry["llms_txt"]
        if _is_http_or_https(url):
            remote_sources.append(entry)
        else:
            local_sources.append(entry)

    # Let's verify that all local sources exist
    for entry in local_sources:
        path = entry["llms_txt"]
        abs_path = _normalize_path(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Local file not found: {abs_path}")

    # Parse the domain names in the llms.txt URLs and identify local file paths
    domains = set(extract_domain(entry["llms_txt"]) for entry in remote_sources)
    oauth2_by_domain: dict[str, OAuth2Config] = {}
    oauth2_token_cache: dict[str, OAuth2TokenCacheEntry] = {}

    for entry in remote_sources:
        oauth2_config = entry.get("oauth2")
        if not oauth2_config:
            continue

        missing_fields = [
            field
            for field in ("token_url", "client_id", "client_secret")
            if not oauth2_config.get(field)
        ]
        if missing_fields:
            raise ValueError(
                f"OAuth2 config for {entry['llms_txt']} is missing required fields: "
                + ", ".join(missing_fields)
            )

        domain = extract_domain(entry["llms_txt"])
        existing_config = oauth2_by_domain.get(domain)
        if existing_config and existing_config != oauth2_config:
            raise ValueError(
                f"Multiple OAuth2 configurations found for the same domain: {domain}"
            )
        oauth2_by_domain[domain] = oauth2_config

    # Add additional allowed domains if specified, or set to '*' if we have local files
    if allowed_domains:
        if "*" in allowed_domains:
            domains = {"*"}  # Special marker for allowing all domains
        else:
            domains.update(allowed_domains)

    allowed_local_files = set(
        _normalize_path(entry["llms_txt"]) for entry in local_sources
    )

    @server.tool()
    def list_doc_sources() -> str:
        """List all available documentation sources.

        This is the first tool you should call in the documentation workflow.
        It provides URLs to llms.txt files or local file paths that the user has made available.

        Returns:
            A string containing a formatted list of documentation sources with their URLs or file paths
        """
        content = ""
        for entry_ in doc_sources:
            url_or_path = entry_["llms_txt"]

            if _is_http_or_https(url_or_path):
                name = entry_.get("name", extract_domain(url_or_path))
                content += f"{name}\nURL: {url_or_path}\n\n"
            else:
                path = _normalize_path(url_or_path)
                name = entry_.get("name", path)
                content += f"{name}\nPath: {path}\n\n"
        return content

    fetch_docs_description = _get_fetch_description(
        has_local_sources=bool(local_sources)
    )

    @server.tool(description=fetch_docs_description)
    async def fetch_docs(url: str) -> str:
        nonlocal domains, follow_redirects
        url = url.strip()
        # Handle local file paths (either as file:// URLs or direct filesystem paths)
        if not _is_http_or_https(url):
            abs_path = _normalize_path(url)
            if abs_path not in allowed_local_files:
                raise ValueError(
                    f"Local file not allowed: {abs_path}. Allowed files: {allowed_local_files}"
                )
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return markdownify(content)
            except Exception as e:
                return f"Error reading local file: {str(e)}"
        else:
            # Otherwise treat as URL
            if "*" not in domains and not any(
                url.startswith(domain) for domain in domains
            ):
                return (
                    "Error: URL not allowed. Must start with one of the following domains: "
                    + ", ".join(domains)
                )

            try:
                headers = await _get_oauth2_headers(
                    url=url,
                    httpx_client=httpx_client,
                    oauth2_by_domain=oauth2_by_domain,
                    oauth2_token_cache=oauth2_token_cache,
                )
                response = await httpx_client.get(url, timeout=timeout, headers=headers)
                response.raise_for_status()
                content = response.text

                if follow_redirects:
                    # Check for meta refresh tag which indicates a client-side redirect
                    match = re.search(
                        r'<meta http-equiv="refresh" content="[^;]+;\s*url=([^"]+)"',
                        content,
                        re.IGNORECASE,
                    )

                    if match:
                        redirect_url = match.group(1)
                        new_url = urljoin(str(response.url), redirect_url)

                        if "*" not in domains and not any(
                            new_url.startswith(domain) for domain in domains
                        ):
                            return (
                                "Error: Redirect URL not allowed. Must start with one of the following domains: "
                                + ", ".join(domains)
                            )

                        redirect_headers = await _get_oauth2_headers(
                            url=new_url,
                            httpx_client=httpx_client,
                            oauth2_by_domain=oauth2_by_domain,
                            oauth2_token_cache=oauth2_token_cache,
                        )
                        response = await httpx_client.get(
                            new_url, timeout=timeout, headers=redirect_headers
                        )
                        response.raise_for_status()
                        content = response.text

                return markdownify(content)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                return f"Encountered an HTTP error: {str(e)}"

    return server
