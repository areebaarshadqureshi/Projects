"""
Unit tests for tools/job_search_tool.py.

Mocks the DDGS class entirely -- no real web search calls. Patched at
"tools.job_search_tool.DDGS" (where job_search_tool.py looks the name up
after `from ddgs import DDGS`), not "ddgs.DDGS" (where the class is
defined) -- patching the definition site would have no effect here,
since job_search_tool already holds its own reference to the class.
"""

from unittest.mock import patch, MagicMock
from tools.job_search_tool import search_job_postings, job_search_tool


@patch("tools.job_search_tool.DDGS")
def test_search_job_postings_returns_parsed_results(mock_ddgs_cls_target):
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {"title": "Data Scientist - Karachi", "body": "Looking for...", "href": "https://example.com/job1"},
    ]
    mock_ddgs_cls_target.return_value.__enter__.return_value = mock_instance
    mock_ddgs_cls_target.return_value.__exit__.return_value = False

    result = search_job_postings("Data Scientist Karachi")

    assert len(result) == 1
    assert result[0]["title"] == "Data Scientist - Karachi"
    assert result[0]["snippet"] == "Looking for..."
    assert result[0]["url"] == "https://example.com/job1"


@patch("tools.job_search_tool.DDGS")
def test_search_job_postings_passes_num_results_to_ddgs(mock_ddgs_cls_target):
    """
    Asserts the actual call DDGS receives, same principle as asserting
    request params on the GitHub tools -- proves num_results is wired
    through, not silently ignored or hardcoded.
    """
    mock_instance = MagicMock()
    mock_instance.text.return_value = []
    mock_ddgs_cls_target.return_value.__enter__.return_value = mock_instance
    mock_ddgs_cls_target.return_value.__exit__.return_value = False

    search_job_postings("ML internship Karachi", num_results=5)

    mock_instance.text.assert_called_once_with("ML internship Karachi", max_results=5)


@patch("tools.job_search_tool.DDGS")
def test_job_search_tool_structured_tool_invoke(mock_ddgs_cls_target):
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {"title": "AI Intern", "body": "snippet", "href": "https://example.com/job2"},
    ]
    mock_ddgs_cls_target.return_value.__enter__.return_value = mock_instance
    mock_ddgs_cls_target.return_value.__exit__.return_value = False

    result = job_search_tool.invoke({"query": "AI Intern Pakistan"})

    assert len(result) == 1
    assert result[0]["title"] == "AI Intern"


def test_job_search_input_rejects_out_of_range_num_results():
    from pydantic import ValidationError
    from tools.job_search_tool import JobSearchInput

    for bad_value in (0, 11, -1):
        try:
            JobSearchInput(query="test", num_results=bad_value)
            assert False, f"num_results={bad_value} should have raised ValidationError"
        except ValidationError:
            pass
