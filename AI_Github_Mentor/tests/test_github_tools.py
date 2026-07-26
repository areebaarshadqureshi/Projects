"""
Unit tests for tools/github_repo_tool.py.

Mocks requests.get entirely -- no real GitHub API calls, no rate limits,
no network dependency, so this suite is safe to run in CI or offline.
"""

from unittest.mock import patch, MagicMock
from tools.github_repo_tool import fetch_all_repo_data, github_repo_tool


def _mock_response(json_data, status_code=200, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400 and status_code not in (404, 409):
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


@patch("utils.github_api_client.requests.get")
def test_fetch_all_repo_data_returns_enriched_repos(mock_get):
    repos_response = _mock_response([
        {"name": "repo-a", "description": "desc a", "language": "Python",
         "topics": ["ml"], "html_url": "https://github.com/user/repo-a"},
    ])
    readme_response = _mock_response({}, text="# Repo A\nThis does X.")
    commits_response = _mock_response([{"sha": "abc"}] * 3)
    contents_response = _mock_response([])

    # get_user_repos -> get_repo_readme -> get_repo_commits -> get_repo_root_contents, in that call order
    mock_get.side_effect = [repos_response, readme_response, commits_response, contents_response]

    result = fetch_all_repo_data("some-user")

    assert len(result) == 1
    assert result[0]["name"] == "repo-a"
    assert result[0]["readme"] == "# Repo A\nThis does X."
    assert result[0]["commit_count"] == 3


@patch("utils.github_api_client.requests.get")
def test_fetch_all_repo_data_handles_missing_readme(mock_get):
    repos_response = _mock_response([
        {"name": "repo-b", "description": None, "language": None,
         "topics": [], "html_url": "https://github.com/user/repo-b"},
    ])
    readme_404 = _mock_response({}, status_code=404)
    commits_response = _mock_response([])
    contents_response = _mock_response([])

    mock_get.side_effect = [repos_response, readme_404, commits_response, contents_response]

    result = fetch_all_repo_data("some-user")

    assert result[0]["readme"] == ""
    assert result[0]["description"] == ""  # None coerced to ""


@patch("utils.github_api_client.requests.get")
def test_fetch_all_repo_data_passes_max_repos_to_api(mock_get):
    """
    The whole point of max_repos is to cap and sort server-side, not fetch
    everything and slice in Python. This test asserts the actual request
    params GitHub receives, not just the returned list length -- a bug that
    slices client-side after fetching 100 repos would pass a length-only
    test but still burn the full rate-limit cost.
    """
    repos_response = _mock_response([
        {"name": "repo-a", "description": "", "language": "Python",
         "topics": [], "html_url": "https://github.com/user/repo-a"},
    ])
    readme_response = _mock_response({}, text="")
    commits_response = _mock_response([])
    contents_response = _mock_response([])
    mock_get.side_effect = [repos_response, readme_response, commits_response, contents_response]

    fetch_all_repo_data("some-user", max_repos=5)

    repos_call = mock_get.call_args_list[0]
    assert repos_call.kwargs["params"] == {
        "per_page": 5,
        "sort": "updated",
        "direction": "desc",
    }


@patch("utils.github_api_client.requests.get")
def test_fetch_all_repo_data_defaults_max_repos_to_20(mock_get):
    repos_response = _mock_response([])
    mock_get.return_value = repos_response

    fetch_all_repo_data("some-user")

    repos_call = mock_get.call_args_list[0]
    assert repos_call.kwargs["params"]["per_page"] == 20


@patch("utils.github_api_client.requests.get")
def test_github_repo_tool_structured_tool_invoke(mock_get):
    repos_response = _mock_response([
        {"name": "repo-c", "description": "d", "language": "Python",
         "topics": [], "html_url": "https://github.com/user/repo-c"},
    ])
    readme_response = _mock_response({}, text="readme text")
    commits_response = _mock_response([])
    contents_response = _mock_response([])
    mock_get.side_effect = [repos_response, readme_response, commits_response, contents_response]

    result = github_repo_tool.invoke({"username": "some-user"})

    assert len(result) == 1
    assert result[0]["name"] == "repo-c"


def test_repo_fetch_input_rejects_out_of_range_max_repos():
    """
    max_repos=0 or max_repos=101 should never reach fetch_all_repo_data --
    Pydantic should reject it at the schema boundary. This is the actual
    payoff of wrapping this as a StructuredTool with a validated
    args_schema instead of a plain function.
    """
    from pydantic import ValidationError
    from tools.github_repo_tool import RepoFetchInput

    for bad_value in (0, 101, -5):
        try:
            RepoFetchInput(username="some-user", max_repos=bad_value)
            assert False, f"max_repos={bad_value} should have raised ValidationError"
        except ValidationError:
            pass


# --- possible_monorepo detection ---

def test_looks_like_monorepo_flags_repo_with_several_project_dirs():
    from tools.github_repo_tool import _looks_like_monorepo
    contents = [
        {"type": "dir", "name": "karachi-real-estate-fraud-detection"},
        {"type": "dir", "name": "urdu-emotion-detector"},
        {"type": "dir", "name": "psl-cricket-dashboard"},
        {"type": "file", "name": "README.md"},
    ]
    assert _looks_like_monorepo(contents) is True


def test_looks_like_monorepo_ignores_common_non_project_dirs():
    """
    A normal single-project repo with src/, tests/, docs/ should NOT be
    flagged -- this is the false-positive case the exclusion list exists
    to prevent.
    """
    from tools.github_repo_tool import _looks_like_monorepo
    contents = [
        {"type": "dir", "name": "src"},
        {"type": "dir", "name": "tests"},
        {"type": "dir", "name": "docs"},
        {"type": "file", "name": "README.md"},
    ]
    assert _looks_like_monorepo(contents) is False


def test_looks_like_monorepo_false_for_empty_or_missing_contents():
    from tools.github_repo_tool import _looks_like_monorepo
    assert _looks_like_monorepo([]) is False


@patch("utils.github_api_client.requests.get")
def test_fetch_all_repo_data_includes_possible_monorepo_flag(mock_get):
    repos_response = _mock_response([
        {"name": "repo-a", "description": "", "language": "Python",
         "topics": [], "html_url": "https://github.com/user/repo-a"},
    ])
    readme_response = _mock_response({}, text="")
    commits_response = _mock_response([])
    contents_response = _mock_response([
        {"type": "dir", "name": "sub1"}, {"type": "dir", "name": "sub2"}, {"type": "dir", "name": "sub3"},
    ])
    mock_get.side_effect = [repos_response, readme_response, commits_response, contents_response]

    result = fetch_all_repo_data("some-user")

    assert result[0]["possible_monorepo"] is True
