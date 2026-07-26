"""
Tests for scripts/fetch_job_postings.py's run(). Did not exist in v1.

Mocks job_search_tool and extract_job_requirement entirely -- no
real web search or LLM calls. JOB_REQUIREMENTS_DIR and ROLE_QUERIES are
both monkeypatched: JOB_REQUIREMENTS_DIR to a tmp_path (so nothing
touches the real data/ directory), and ROLE_QUERIES to a single entry
per test, so each test's assertions aren't diluted across all 4 real
roles run() normally processes.
"""

import json
from unittest.mock import patch
import scripts.fetch_job_postings as fetch_job_postings
from chains.job_extraction_chain import ExtractedJobRequirement


def _fake_extracted(role="Data Scientist", url="https://example.com/1", karachi=True):
    return ExtractedJobRequirement(
        role=role, seniority="entry-level", required_skills=["Python"],
        nice_to_have=[], source_url=url, notes="note", is_karachi_relevant=karachi,
    )


@patch("scripts.fetch_job_postings.time.sleep")
@patch("scripts.fetch_job_postings.extract_job_requirement")
@patch("scripts.fetch_job_postings.job_search_tool")
def test_run_skips_duplicate_urls_across_runs(mock_search, mock_extract, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_job_postings, "JOB_REQUIREMENTS_DIR", tmp_path)
    monkeypatch.setattr(fetch_job_postings, "ROLE_QUERIES", {"data_scientist_karachi": "Data Scientist Karachi"})

    existing_file = tmp_path / "data_scientist_karachi.json"
    existing_file.write_text(json.dumps([_fake_extracted(url="https://example.com/already-seen").model_dump()]))

    mock_search.invoke.return_value = [
        {"title": "t", "snippet": "s", "url": "https://example.com/already-seen"},  # duplicate
        {"title": "t2", "snippet": "s2", "url": "https://example.com/new-one"},      # new
    ]
    mock_extract.return_value = _fake_extracted(url="https://example.com/new-one")

    fetch_job_postings.run(llm=None)

    assert mock_extract.call_count == 1  # only called for the non-duplicate result
    saved = json.loads(existing_file.read_text())
    assert len(saved) == 2  # 1 pre-existing + 1 newly added
    assert any(e["source_url"] == "https://example.com/new-one" for e in saved)


@patch("scripts.fetch_job_postings.time.sleep")
@patch("scripts.fetch_job_postings.extract_job_requirement")
@patch("scripts.fetch_job_postings.job_search_tool")
def test_run_filters_non_karachi_relevant_results(mock_search, mock_extract, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_job_postings, "JOB_REQUIREMENTS_DIR", tmp_path)
    monkeypatch.setattr(fetch_job_postings, "ROLE_QUERIES", {"data_scientist_karachi": "Data Scientist Karachi"})

    mock_search.invoke.return_value = [{"title": "t", "snippet": "s", "url": "https://example.com/x"}]
    mock_extract.return_value = _fake_extracted(url="https://example.com/x", karachi=False)

    fetch_job_postings.run(llm=None)

    saved_file = tmp_path / "data_scientist_karachi.json"
    saved = json.loads(saved_file.read_text()) if saved_file.exists() else []
    assert saved == []  # non-Karachi-relevant result dropped for a Karachi-targeted query


@patch("scripts.fetch_job_postings.time.sleep")
@patch("scripts.fetch_job_postings.extract_job_requirement")
@patch("scripts.fetch_job_postings.job_search_tool")
def test_run_keeps_non_karachi_results_for_nlp_query(mock_search, mock_extract, mock_sleep, tmp_path, monkeypatch):
    """
    The source comments explicitly carve out an exception for the NLP
    role since there are few local postings -- worth proving that
    exception actually fires, not just trusting the comment.
    """
    monkeypatch.setattr(fetch_job_postings, "JOB_REQUIREMENTS_DIR", tmp_path)
    monkeypatch.setattr(fetch_job_postings, "ROLE_QUERIES", {"nlp_engineer_karachi": "NLP Engineer Pakistan"})

    mock_search.invoke.return_value = [{"title": "t", "snippet": "s", "url": "https://example.com/nlp1"}]
    mock_extract.return_value = _fake_extracted(role="NLP Engineer", url="https://example.com/nlp1", karachi=False)

    fetch_job_postings.run(llm=None)

    saved = json.loads((tmp_path / "nlp_engineer_karachi.json").read_text())
    assert len(saved) == 1


@patch("scripts.fetch_job_postings.time.sleep")
@patch("scripts.fetch_job_postings.extract_job_requirement")
@patch("scripts.fetch_job_postings.job_search_tool")
def test_run_continues_after_one_extraction_error(mock_search, mock_extract, mock_sleep, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_job_postings, "JOB_REQUIREMENTS_DIR", tmp_path)
    monkeypatch.setattr(fetch_job_postings, "ROLE_QUERIES", {"data_scientist_karachi": "Data Scientist Karachi"})

    mock_search.invoke.return_value = [
        {"title": "bad", "snippet": "s", "url": "https://example.com/bad"},
        {"title": "good", "snippet": "s", "url": "https://example.com/good"},
    ]
    mock_extract.side_effect = [
        Exception("model returned invalid JSON"),
        _fake_extracted(url="https://example.com/good"),
    ]

    fetch_job_postings.run(llm=None)  # should not raise

    saved = json.loads((tmp_path / "data_scientist_karachi.json").read_text())
    assert len(saved) == 1
    assert saved[0]["source_url"] == "https://example.com/good"
