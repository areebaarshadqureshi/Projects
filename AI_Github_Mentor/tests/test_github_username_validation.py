"""
Tests for looks_like_url, is_valid_github_profile_url, and
check_username_exists -- moved into utils/github_api_client.py from
app/streamlit_app.py in v2. None of these had tests before, since they
were only reachable through a running Streamlit app.
"""

from unittest.mock import patch, MagicMock
from utils.github_api_client import (
    looks_like_url,
    is_valid_github_profile_url,
    check_username_exists,
)


def test_looks_like_url_detects_http_and_www():
    assert looks_like_url("https://github.com/octocat")
    assert looks_like_url("www.github.com/octocat")
    assert looks_like_url("http://example.com")


def test_looks_like_url_rejects_bare_username():
    assert not looks_like_url("octocat")


def test_is_valid_github_profile_url_accepts_bare_profile():
    assert is_valid_github_profile_url("https://github.com/octocat")
    assert is_valid_github_profile_url("github.com/octocat")
    assert is_valid_github_profile_url("https://www.github.com/octocat/")


def test_is_valid_github_profile_url_rejects_repo_path():
    assert not is_valid_github_profile_url("https://github.com/octocat/hello-world")


def test_is_valid_github_profile_url_rejects_non_github_domain():
    assert not is_valid_github_profile_url("https://octocat.github.io")
    assert not is_valid_github_profile_url("https://gist.github.com/octocat")


@patch("utils.github_api_client.requests.get")
def test_check_username_exists_returns_true_on_200(mock_get):
    mock_get.return_value = MagicMock(status_code=200)
    assert check_username_exists("octocat") is True


@patch("utils.github_api_client.requests.get")
def test_check_username_exists_returns_false_on_404(mock_get):
    mock_get.return_value = MagicMock(status_code=404)
    assert check_username_exists("definitely-not-a-real-user-xyz") is False


@patch("utils.github_api_client.requests.get")
def test_check_username_exists_fails_open_on_network_error(mock_get):
    """
    A network hiccup on this quick pre-check should NOT block the user --
    the real audit call right after will surface a proper GitHubAPIError
    if GitHub is genuinely unreachable. This is a deliberate fail-open,
    not a bug -- worth a test specifically so nobody "fixes" it later
    into fail-closed without realizing it was intentional.
    """
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError("network down")
    assert check_username_exists("octocat") is True
