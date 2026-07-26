"""
tools/job_search_tool.py

Wraps DuckDuckGo search (via the `ddgs` package) as a LangChain StructuredTool.
No API key or signup required -- replaces the old Google Custom Search
implementation, which is closed to new customers and shutting down
entirely on Jan 1, 2027.

Free, but unofficial: DuckDuckGo may rate-limit or throttle bursts of
requests. Don't loop this tightly; add a short delay between calls if
you see empty results.
"""

from ddgs import DDGS
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class JobSearchInput(BaseModel):
    query: str = Field(description="Search query, e.g. 'Data Scientist Karachi job requirements'")
    num_results: int = Field(
        default=10,
        ge=1,
        le=10,
        description=(
            "Number of results to fetch. Capped at 10 -- DuckDuckGo's free "
            "text search doesn't reliably return more than that per call."
        ),
    )


def search_job_postings(query: str, num_results: int = 10) -> list[dict]:
    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=num_results))

    results = []
    for item in raw_results:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("body", ""),
            "url": item.get("href", ""),
        })
    return results


job_search_tool = StructuredTool.from_function(
    func=search_job_postings,
    name="search_job_postings",
    description="Searches the web for job postings matching a query",
    args_schema=JobSearchInput,
)
