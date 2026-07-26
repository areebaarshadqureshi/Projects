"""
scripts/fetch_job_postings.py

Runs the full pipeline: search for each target role, extract each
result into structured JSON, and append to data/job_requirements/*.json.

Run this occasionally (e.g. once a week) to grow the dataset over time,
not in a tight loop. This uses tools.job_search_tool, which is backed
by DuckDuckGo (via the `ddgs` package) -- free and keyless, but
unofficial, so it may throttle or return empty results under heavy or
rapid use. Don't loop this tightly; the 1-second sleep between LLM
extraction calls below also gives search requests some natural spacing.

Usage (from the project root):
    python -m scripts.run_fetch_job_postings
"""

import json
import time

from tools.job_search_tool import job_search_tool
from chains.job_extraction_chain import extract_job_requirement
from configs.settings import DATA_DIR  # single source of truth for the path,
                                         # matches what build_job_index.py uses

JOB_REQUIREMENTS_DIR = DATA_DIR / "job_requirements"
JOB_REQUIREMENTS_DIR.mkdir(parents=True, exist_ok=True)

ROLE_QUERIES = {
    "data_scientist_karachi": "Data Scientist Karachi job requirements skills",
    "data_analyst_karachi": "Data Analyst Karachi job requirements skills",
    "ml_engineer_karachi": "Machine Learning Engineer Karachi job requirements skills",
    "nlp_engineer_karachi": "NLP Engineer Pakistan job requirements skills",
}


def load_existing(file_path) -> list[dict]:
    if file_path.exists() and file_path.stat().st_size > 0:
        return json.loads(file_path.read_text())
    return []


def already_have_url(existing: list[dict], url: str) -> bool:
    return any(entry.get("source_url") == url for entry in existing)


def run(llm):
    print(f"Writing job requirement files to: {JOB_REQUIREMENTS_DIR.resolve()}")

    for filename_stem, query in ROLE_QUERIES.items():
        file_path = JOB_REQUIREMENTS_DIR / f"{filename_stem}.json"
        existing = load_existing(file_path)

        print(f"\nSearching: {query}")
        results = job_search_tool.invoke({"query": query, "num_results": 10})
        print(f"  Search returned {len(results)} results")

        new_entries = []
        for result in results:
            if already_have_url(existing, result["url"]):
                continue  # skip duplicates across runs
            try:
                extracted = extract_job_requirement(llm, result)
            except Exception as error:
                print(f"  Skipped one result due to extraction error: {error}")
                continue

            if not extracted.is_karachi_relevant and "nlp" not in filename_stem:
                continue  # keep the dataset market-relevant, except NLP which has few local postings

            new_entries.append(extracted.model_dump(exclude={"is_karachi_relevant"}))
            time.sleep(1)  # be reasonable with LLM call pacing

        combined = existing + new_entries
        file_path.write_text(json.dumps(combined, indent=2))
        print(f"  Added {len(new_entries)} new entries to {file_path.name} (total: {len(combined)})")


if __name__ == "__main__":
    raise SystemExit(
        "Run this via: python -m scripts.run_fetch_job_postings\n"
        "(this file needs an llm object passed to run(), it doesn't create one itself)"
    )
