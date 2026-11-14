"""Public web tools use sourced results and never reach private networks."""
import asyncio
import json
import socket
from types import SimpleNamespace

import httpx
import pytest

from lucifer.tools import ToolContext, ToolRegistry
from lucifer.tools import web
from lucifer.tools.workspace import ToolError, ToolPolicyError


@pytest.fixture(autouse=True)
def clear_web_state(monkeypatch):
    web._CACHE.clear()
    web._LAST_SEARCH.clear()
    monkeypatch.delenv("LUCIFER_SEARXNG_URL", raising=False)


@pytest.fixture
def context(tmp_path):
    return ToolContext(tmp_path)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/", "http://10.0.0.2/", "http://[::ffff:127.0.0.1]/", "http://localhost/",
    "http://machine.local/", "https://user:password@example.com/", "https://example.com:8000/",
    "https://example.com/\nsecret", "http://2130706433/", "https://example.invalid/",
])
def test_public_url_blocks_nonpublic_credentials_and_ambiguous_targets(url):
    with pytest.raises(ToolPolicyError):
        web.public_url(url)


@pytest.mark.asyncio
async def test_dns_resolution_rejects_mixed_public_and_private_answers(monkeypatch):
    async def addresses(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
                for ip in ["93.184.216.34", "127.0.0.1"]]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", addresses)
    with pytest.raises(ToolPolicyError, match="private"):
        await web._public_addresses("https://example.com")


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return original(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(web.httpx, "AsyncClient", client)
    async def public_addresses(url):
        web.public_url(url)
        return ["93.184.216.34"]
    monkeypatch.setattr(web, "_public_addresses", public_addresses)


@pytest.mark.asyncio
async def test_fetch_connects_to_validated_ip_with_original_host_and_tls_sni(monkeypatch):
    def handler(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "authorization" not in request.headers
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="public evidence")
    mock_http(monkeypatch, handler)
    result = await web.fetch_public("https://example.com/evidence")
    assert result.url == "https://example.com/evidence"
    assert result.body == b"public evidence"


@pytest.mark.asyncio
async def test_redirect_is_revalidated_before_second_connection(monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/secret"})
    mock_http(monkeypatch, handler)
    with pytest.raises(ToolPolicyError):
        await web.fetch_public("https://example.com/")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_oversized_public_download_is_rejected(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 200))
    with pytest.raises(ToolError, match="limit"):
        await web.fetch_public("https://example.com", max_bytes=100)


@pytest.mark.asyncio
async def test_cancelling_waiting_download_closes_owned_request(monkeypatch):
    started, finished, cancel = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def blocked(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
    monkeypatch.setattr(web, "_fetch", blocked)
    pending = asyncio.create_task(web.fetch_public("https://example.com", cancel=cancel))
    await started.wait()
    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 1)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_search_fallback_returns_actual_sourced_results_and_uses_cache(monkeypatch, context):
    calls = []
    async def fetch(url, **kwargs):
        calls.append(url)
        if "duckduckgo" in url:
            raise ToolError("HTTP 202 challenge")
        return web.FetchResult(b'''<rss><channel><title>Bing: python docs</title><item><title>Python Docs</title>
          <link>https://docs.python.org/3/</link><description>Language reference</description>
          </item><item><title>Local trap</title><link>http://127.0.0.1/</link></item></channel></rss>''',
                               "application/rss+xml", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    result = await web.execute_web("web_search", {"query": "python docs"}, context)
    assert result["provider"] == "bing_rss" and result["status"] == "ok"
    assert result["results"] == [{"title": "Python Docs", "url": "https://docs.python.org/3/",
                                   "snippet": "Language reference"}]
    assert result["untrusted_source"] is True
    again = await web.execute_web("web_search", {"query": "python docs"}, context)
    assert again["cached"] and len(calls) == 2


@pytest.mark.asyncio
async def test_unavailable_search_reports_failures_without_fabricating_content(monkeypatch, context):
    async def fail(*args, **kwargs):
        raise ToolError("HTTP 429 rate limit")
    monkeypatch.setattr(web, "fetch_public", fail)
    result = await web.execute_web("web_search", {"query": "unlikely current fact"}, context)
    assert result["results"] == [] and result["status"] == "unavailable"
    assert len(result["providers"]) == 3
    assert all("429" in failure["error"] for failure in result["providers"])


@pytest.mark.asyncio
async def test_unrelated_engine_results_are_rejected_and_technical_fallback_is_disclosed(monkeypatch, context):
    async def fetch(url, **kwargs):
        if "duckduckgo" in url:
            raise ToolError("HTTP 202 challenge")
        if "bing.com" in url:
            return web.FetchResult(b'''<rss><channel><title>Bing: SearXNG Search API</title>
              <item><title>Bank account</title><link>https://example.com/banking</link>
              <description>Use our banking API.</description></item></channel></rss>''',
                                   "application/rss+xml", url)
        assert "api.github.com/search/repositories" in url
        return web.FetchResult(json.dumps({"items": [{
            "full_name": "searxng/searxng", "html_url": "https://github.com/searxng/searxng",
            "description": "SearXNG public metasearch engine", "homepage": "https://docs.searxng.org/",
        }]}).encode(), "application/json", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    result = await web.execute_web("web_search", {"query": "SearXNG Search API"}, context)
    assert result["provider"] == "github_repositories"
    assert result["coverage"] == "public_software_repositories"
    assert "General web engines" in result["warning"]
    assert result["relevance_checked"] is True
    assert [row["url"] for row in result["results"]] == [
        "https://github.com/searxng/searxng", "https://docs.searxng.org/",
    ]


def test_query_relevance_and_site_filters_reject_wrong_topics_and_deceptive_domains():
    rows = [
        {"title": "SearXNG API", "url": "https://docs.searxng.org/dev/search_api.html", "snippet": "Parameters"},
        {"title": "Bank API", "url": "https://example.com/", "snippet": "Online banking"},
        {"title": "SearXNG API", "url": "https://docs.searxng.org.example.com/", "snippet": "Parameters"},
    ]
    accepted = web._relevant_results(rows, "site:docs.searxng.org SearXNG Search API")
    assert accepted == rows[:1]


def test_search_response_must_identify_the_requested_query():
    response = web.FetchResult(b"<rss><channel><title>Bing: weather</title></channel></rss>", "application/rss+xml", "https://www.bing.com/search")
    with pytest.raises(ToolError, match="requested query"):
        web._parse_search("bing_rss", response, "SearXNG")


@pytest.mark.asyncio
async def test_page_reader_extracts_text_and_pages_cached_source_without_executing_scripts(monkeypatch, context):
    calls = []
    async def fetch(url, **kwargs):
        calls.append(url)
        return web.FetchResult(b"<html><title>Source</title><script>STEAL SECRET</script><body>"
                               + b"<p>verified paragraph</p>" * 100 + b"</body></html>", "text/html", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    first = await web.execute_web("read_web_page", {"url": "https://example.com", "max_chars": 120}, context)
    assert first["title"] == "Source" and "STEAL" not in first["content"]
    assert first["next_start_char"] == 120 and first["truncated"]
    second = await web.execute_web("read_web_page", {"url": "https://example.com", "start_char": 120,
                                                       "max_chars": 120}, context)
    assert second["start_char"] == 120 and second["cached"] and len(calls) == 1


@pytest.mark.asyncio
async def test_page_reader_returns_actual_resolved_public_links_without_fetching_them(monkeypatch, context):
    calls = []
    async def fetch(url, **kwargs):
        calls.append(url)
        return web.FetchResult(b'''<html><base href="/dev/"><title>Documentation</title>
          <a href="search_api.html">Search <b>API</b></a>
          <a href="https://docs.example.org/guide">External guide</a>
          <a href="http://127.0.0.1/private">Private trap</a>
          <a href="javascript:alert(1)">Script trap</a>
          <a href="#heading">Current page section</a></html>''', "text/html", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    result = await web.execute_web("read_web_page", {"url": "https://example.com"}, context)
    assert result["links"] == [
        {"title": "Search API", "url": "https://example.com/dev/search_api.html"},
        {"title": "External guide", "url": "https://docs.example.org/guide"},
    ]
    assert len(calls) == 1 and result["links_truncated"] is False


def test_page_link_metadata_stays_below_eight_kibibytes_and_marks_incomplete_coverage():
    html = "".join(f'<a href="https://example.com/{number}/' + "long/" * 80 + '">Page title</a>'
                   for number in range(100))
    links, truncated = web._page_links(web.FetchResult(html.encode(), "text/html", "https://example.com"))
    assert len(links) <= 30 and len(json.dumps(links, ensure_ascii=False).encode()) <= 8192
    assert truncated is True


@pytest.mark.asyncio
async def test_page_link_pagination_reaches_later_source_links_using_cached_page(monkeypatch, context):
    calls = []
    async def fetch(url, **kwargs):
        calls.append(url)
        html = "".join(f'<a href="/page{number}.html">Page {number}</a>' for number in range(65))
        return web.FetchResult(html.encode(), "text/html", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    first = await web.execute_web("read_web_page", {"url": "https://example.com"}, context)
    assert first["next_link_offset"] == 30 and first["total_links"] == 65
    second = await web.execute_web("read_web_page", {"url": "https://example.com", "link_offset": 30}, context)
    assert second["links"][0]["url"] == "https://example.com/page30.html"
    assert second["next_link_offset"] == 60 and second["cached"]
    last = await web.execute_web("read_web_page", {"url": "https://example.com", "link_offset": 60}, context)
    assert len(last["links"]) == 5 and last["next_link_offset"] is None
    assert last["links_truncated"] is False and len(calls) == 1
    filtered = await web.execute_web("read_web_page", {"url": "https://example.com", "link_query": "Page 64"}, context)
    assert filtered["links"] == [{"title": "Page 64", "url": "https://example.com/page64.html"}]
    assert filtered["total_links"] == 1 and filtered["source_total_links"] == 65
    assert filtered["cached"] and len(calls) == 1


@pytest.mark.asyncio
async def test_web_page_reader_rejects_binary_document_without_decoding_it(monkeypatch, context):
    async def fetch(url, **kwargs):
        return web.FetchResult(b"%PDF-1.7 arbitrary bytes", "application/pdf", url)
    monkeypatch.setattr(web, "fetch_public", fetch)
    result = await web.execute_web("read_web_page", {"url": "https://example.com/doc.pdf"}, context)
    assert result["status"] == "unavailable" and "attach" in result["error"]


def test_duckduckgo_html_parser_unwraps_search_links():
    response = web.FetchResult(b'''<div><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F">Python <b>Docs</b></a>
                                <a class="result__snippet">Read the language reference.</a></div>''', "text/html", "https://html.duckduckgo.com/")
    assert web._parse_search("duckduckgo", response) == [
        {"title": "Python Docs", "url": "https://docs.python.org/3/", "snippet": "Read the language reference."}]


def test_registry_exposes_web_reads_without_command_or_mcp_grants(context):
    registry = ToolRegistry()
    names = {item["function"]["name"] for item in registry.specs(context)}
    assert {"web_search", "read_web_page"} <= names
    assert not registry.prepare("web_search", {"query": "documentation"}, context).requires_approval
    disabled = SimpleNamespace(**context.__dict__)
    disabled.allow_web = False
    assert "web_search" not in {item["function"]["name"] for item in registry.specs(disabled)}
