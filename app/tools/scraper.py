"""
URL scraping tool: fetches a URL, strips HTML, returns clean text.
"""

import re
import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_core.tools import tool
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_CONTENT_CHARS = 4000  # keep context window manageable

BLOCKED_EXTENSIONS = {".pdf", ".zip", ".mp4", ".mp3", ".png", ".jpg", ".jpeg"}


def _is_scrapable(url: str) -> bool:
    return not any(url.lower().endswith(ext) for ext in BLOCKED_EXTENSIONS)


def _clean_html(html: str) -> str:
    """Strip tags, collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style/nav/footer noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CONTENT_CHARS]


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
async def _fetch_url(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ResearchAgent/1.0; +https://yoursite.com/bot)"
        )
    }
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


@tool
async def scrape_url(url: str) -> str:
    """
    Fetch and extract the main text content from a URL.

    Args:
        url: The full URL to scrape (must start with http:// or https://).

    Returns:
        Cleaned plain-text content from the page, truncated to avoid token overuse.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url}. Must start with http:// or https://"

    if not _is_scrapable(url):
        return f"Skipping non-scrapable URL: {url}"

    try:
        html = await _fetch_url(url)
        content = _clean_html(html)
        logger.info("scrape_complete", url=url, chars=len(content))
        return content or "No extractable text content found."
    except httpx.HTTPStatusError as e:
        logger.warning("scrape_http_error", url=url, status=e.response.status_code)
        return f"HTTP {e.response.status_code} error fetching {url}"
    except Exception as e:
        logger.error("scrape_failed", url=url, error=str(e))
        return f"Failed to scrape {url}: {str(e)}"
