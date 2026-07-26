"""
Unit tests for tools/github_issue_tool.py.

Mocks requests.get entirely -- no real GitHub API calls, no network
dependency, safe to run offline or in CI.
"""

from unittest.mock import patch, MagicMock
from tools.github_issue_tool import github_issue_tool


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@patch("utils.github_api_client.requests.get")
def test_github_issue_tool_structured_tool_invoke(mock_get):
    issues_response = _mock_response({
        "items": [
            {"title": "Fix bug", "html_url": "https://github.com/x/y/issues/1",
             "repository_url": "https://api.github.com/repos/x/y"},
        ]
    })
    mock_get.return_value = issues_response

    result = github_issue_tool.invoke({"language": "Python"})

    assert len(result) == 1
    assert result[0]["title"] == "Fix bug"


@patch("utils.github_api_client.requests.get")
def test_github_issue_tool_passes_max_issues_to_api(mock_get):
    """
    Same principle as the repo tool's max_repos test: assert the actual
    request params sent to GitHub, not just the returned list -- proves
    max_issues is wired through to per_page, not silently ignored.
    """
    mock_get.return_value = _mock_response({"items": []})

    github_issue_tool.invoke({"language": "JavaScript", "max_issues": 25})

    call = mock_get.call_args
    assert call.kwargs["params"]["per_page"] == 25
    assert "language:JavaScript" in call.kwargs["params"]["q"]


@patch("utils.github_api_client.requests.get")
def test_github_issue_tool_defaults_max_issues_to_10(mock_get):
    mock_get.return_value = _mock_response({"items": []})

    github_issue_tool.invoke({"language": "Python"})

    call = mock_get.call_args
    assert call.kwargs["params"]["per_page"] == 10


def test_issue_search_input_rejects_out_of_range_max_issues():
    from pydantic import ValidationError
    from tools.github_issue_tool import IssueSearchInput

    for bad_value in (0, 31, -1):
        try:
            IssueSearchInput(language="Python", max_issues=bad_value)
            assert False, f"max_issues={bad_value} should have raised ValidationError"
        except ValidationError:
            pass
