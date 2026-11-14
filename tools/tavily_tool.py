import os
import logging
from typing import Optional
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

logger = logging.getLogger(__name__)

_cached_client: Optional[TavilyClient] = None
_cached_key: Optional[str] = None


def _get_client() -> Optional[TavilyClient]:
    """Dynamically get or instantiate the TavilyClient if the API key is set."""
    global _cached_client, _cached_key
    current_key = os.getenv("TAVILY_API_KEY")

    if not current_key:
        return None

    if _cached_client is None or current_key != _cached_key:
        _cached_key = current_key
        _cached_client = TavilyClient(api_key=current_key)

    return _cached_client


def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily and return readable Markdown.
    Handles missing keys, query truncation, API errors and duplicate URLs.
    """
    client = _get_client()
    if not client:
        return "Search unavailable: TAVILY_API_KEY .env file mein set nahi hai."

    if not isinstance(query, str) or not query.strip():
        return "Search failed: Please provide a valid search query."

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5

    max_results = max(1, min(max_results, 10))
    query = query.strip()

    # Guard against Tavily ~400 character limit to prevent HTTP 400 Bad Request
    if len(query) > 380:
        query = query[:380].rsplit(" ", 1)[0]

    try:
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                include_answer=True,
                timeout=12,
            )
        except TypeError:
            # Compatibility with older tavily-python versions without timeout/include_answer
            response = client.search(
                query=query,
                max_results=max_results,
            )

        if not isinstance(response, dict):
            return "Search failed: Invalid response received from search provider."

        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            return "Search failed: Unexpected search results format."

    except Exception as exc:
        err_msg = str(exc).lower()
        logger.exception("Tavily search request failed: %s", exc)

        if "quota" in err_msg or "rate limit" in err_msg or "usage" in err_msg:
            return "Search unavailable: Tavily API quota limit reached. Please check credits."
        if "unauthorized" in err_msg or "api key" in err_msg:
            return "Search unavailable: Invalid TAVILY_API_KEY."

        return "Search failed: Unable to retrieve web results. Please try again."

    results = []
    seen_urls = set()

    # Include synthesized answer if provided by Tavily
    direct_answer = response.get("answer")
    if direct_answer and isinstance(direct_answer, str) and direct_answer.strip():
        results.append(f"💡 **Overview:** {direct_answer.strip()}")

    for result in raw_results:
        if not isinstance(result, dict):
            continue

        title = str(result.get("title") or "Untitled").strip()
        url = str(result.get("url") or "").strip()
        snippet = str(result.get("content") or "").strip()

        if url and url in seen_urls:
            continue

        if url:
            seen_urls.add(url)

        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."

        entry = f"**{title}**"
        if url:
            entry += f"\nURL: {url}"
        if snippet:
            entry += f"\n{snippet}"

        results.append(entry)

        if len(results) >= max_results + (1 if direct_answer else 0):
            break

    if not results:
        return "No relevant search results found."

    formatted_entries = []
    entry_num = 1
    for r in results:
        if r.startswith("💡 **Overview:**"):
            formatted_entries.append(r)
        else:
            formatted_entries.append(f"{entry_num}. {r}")
            entry_num += 1

    return "\n\n".join(formatted_entries)