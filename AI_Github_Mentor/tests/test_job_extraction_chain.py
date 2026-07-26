"""
Tests for chains/job_extraction_chain.py. Did not exist in v1.

The cleaning-related tests moved to test_llm_output_cleaning.py when the
bespoke clean_llm_output_for_pydantic() was replaced with the shared,
more robust clean_llm_json_output() (see utils/llm_output_cleaning.py --
the old version never handled Python True/False/None, which is exactly
what broke contribution_filter_chain in production).
"""

from langchain_core.language_models.fake import FakeListLLM
from chains.job_extraction_chain import extract_job_requirement, ExtractedJobRequirement


def test_extract_job_requirement_returns_structured_result():
    fake_json = ExtractedJobRequirement(
        role="Data Scientist", seniority="entry-level",
        required_skills=["Python", "SQL"], nice_to_have=["Docker"],
        source_url="https://example.com/job1",
        notes="Entry-level role, Karachi-based.",
        is_karachi_relevant=True,
    ).model_dump_json()
    # Wrapped in a markdown fence, as real HF model output often is --
    # proves the cleaning step in the actual chain, not just the helper
    # function in isolation.
    fake_llm = FakeListLLM(responses=[f"```json\n{fake_json}\n```"])

    result = extract_job_requirement(fake_llm, {
        "title": "Data Scientist - Karachi",
        "snippet": "Looking for a junior data scientist...",
        "url": "https://example.com/job1",
    })

    assert isinstance(result, ExtractedJobRequirement)
    assert result.role == "Data Scientist"
    assert result.is_karachi_relevant is True


def test_extract_job_requirement_handles_python_style_booleans():
    """
    Regression test for the exact bug that broke contribution_filter_chain
    in production: a model writing Python's capitalized True/False
    instead of JSON's lowercase true/false.
    """
    raw = '''```json
{
  "role": "NLP Engineer",
  "seniority": "entry-level",
  "required_skills": ["Python"],
  "nice_to_have": [],
  "source_url": "https://example.com/job2",
  "notes": "note",
  "is_karachi_relevant": False
}
```'''
    fake_llm = FakeListLLM(responses=[raw])

    result = extract_job_requirement(fake_llm, {
        "title": "t", "snippet": "s", "url": "https://example.com/job2",
    })

    assert result.is_karachi_relevant is False
