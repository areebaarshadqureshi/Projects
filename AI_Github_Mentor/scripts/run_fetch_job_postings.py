"""
scripts/run_fetch_job_postings.py

Standalone entry point for fetch_job_postings.py -- loads the LLM from
configs/settings.py and runs the pipeline directly, so you don't need
a notebook cell just to kick this off.

Usage:
    python scripts/run_fetch_job_postings.py
"""

from configs.llm_client import get_llm
from scripts.fetch_job_postings import run

if __name__ == "__main__":
    run(get_llm())
