"""
Unit tests for chains/contribution_filter_chain.py.

v1 had zero tests for this chain. These cover: basic filtering, the
is_match structured field actually being required (not optional prose
guessing), max_issues being wired through to the tool call, the removal
of the old hardcoded [:5] slice, and the empty-issues edge case.
"""

from unittest.mock import patch
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.fake import FakeListLLM
from schemas.contribution_match import ContributionMatch
from chains import contribution_filter_chain


def test_filter_issue_returns_contribution_match():
    fake_json = ContributionMatch(
        repo_url="https://api.github.com/repos/x/y",
        issue_title="Fix flaky test",
        issue_url="https://github.com/x/y/issues/1",
        is_match=True,
        relevance_reason="Matches the user's testing experience.",
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = contribution_filter_chain.filter_issue(
        fake_llm, ["Python", "pytest"],
        {"repository_url": "https://api.github.com/repos/x/y",
         "title": "Fix flaky test", "html_url": "https://github.com/x/y/issues/1"},
    )

    assert isinstance(result, ContributionMatch)
    assert result.is_match is True


def test_contribution_match_requires_is_match_field():
    """
    is_match has no default -- proves it's a required structured field,
    not something that silently defaults to False/True if the LLM's
    output happens to omit it.
    """
    from pydantic import ValidationError
    try:
        ContributionMatch(
            repo_url="https://api.github.com/repos/x/y",
            issue_title="t", issue_url="https://github.com/x/y/issues/1",
            relevance_reason="reason given but is_match missing",
        )
        assert False, "should have raised ValidationError without is_match"
    except ValidationError:
        pass


@patch("tools.github_issue_tool.github_issue_tool")
def test_run_contribution_search_passes_max_issues_to_tool(mock_tool):
    mock_tool.invoke.return_value = []

    contribution_filter_chain.run_contribution_search(
        FakeListLLM(responses=[]), ["Python"], "Python", max_issues=7,
    )

    mock_tool.invoke.assert_called_once_with({"language": "Python", "max_issues": 7})


@patch("tools.github_issue_tool.github_issue_tool")
def test_run_contribution_search_filters_all_fetched_issues_not_just_five(mock_tool):
    """
    v1 hardcoded issues[:5] regardless of how many were fetched. This
    proves that's gone -- 8 fetched issues should produce 8 matches.
    """
    mock_tool.invoke.return_value = [
        {"repository_url": f"https://api.github.com/repos/x/repo{i}",
         "title": f"issue {i}", "html_url": f"https://github.com/x/repo{i}/issues/1"}
        for i in range(8)
    ]

    def _route(prompt_value):
        text = prompt_value.to_string()
        # echo back which issue this is, so we can confirm all 8 were processed
        for i in range(8):
            if f"issue {i}" in text:
                return ContributionMatch(
                    repo_url=f"https://api.github.com/repos/x/repo{i}",
                    issue_title=f"issue {i}",
                    issue_url=f"https://github.com/x/repo{i}/issues/1",
                    is_match=True,
                    relevance_reason="ok",
                ).model_dump_json()
        raise AssertionError("unrecognized prompt")

    fake_llm = RunnableLambda(_route)

    results = contribution_filter_chain.run_contribution_search(
        fake_llm, ["Python"], "Python", max_issues=8,
    )

    assert len(results) == 8
    assert {r.issue_title for r in results} == {f"issue {i}" for i in range(8)}


@patch("tools.github_issue_tool.github_issue_tool")
def test_run_contribution_search_handles_no_issues_found(mock_tool):
    mock_tool.invoke.return_value = []

    results = contribution_filter_chain.run_contribution_search(
        FakeListLLM(responses=[]), ["Python"], "Python",
    )

    assert results == []


@patch("tools.github_issue_tool.github_issue_tool")
def test_run_contribution_search_dedups_literal_duplicate_issue_urls(mock_tool):
    """
    Same html_url appearing twice (a literal duplicate) collapses to one.
    """
    mock_tool.invoke.return_value = [
        {"repository_url": "https://api.github.com/repos/x/repo1",
         "title": "Fix pytest deprecation warnings", "html_url": "https://github.com/x/repo1/issues/1"},
        {"repository_url": "https://api.github.com/repos/x/repo1",
         "title": "Fix pytest deprecation warnings", "html_url": "https://github.com/x/repo1/issues/1"},  # literal dup
    ]

    def _route(prompt_value):
        return ContributionMatch(
            repo_url="https://api.github.com/repos/x/repo1",
            issue_title="Fix pytest deprecation warnings",
            issue_url="https://github.com/x/repo1/issues/1",
            is_match=True, relevance_reason="ok",
        ).model_dump_json()

    fake_llm = RunnableLambda(_route)

    results = contribution_filter_chain.run_contribution_search(fake_llm, ["Python"], "Python")

    assert len(results) == 1


@patch("tools.github_issue_tool.github_issue_tool")
def test_run_contribution_search_dedups_same_title_across_different_repos(mock_tool):
    """
    v2: also dedup by title, keeping only the first occurrence -- reversed
    from the earlier design. Evidence from real usage showed same-titled
    issues across different repos are usually forked/cloned copies of the
    same course/template project (e.g. four different accounts' forks of
    "recipe-explorer", each carrying the same auto-generated issue), not
    genuinely distinct opportunities -- showing the same trivial issue 4x
    added no value even though each URL was technically unique.
    """
    mock_tool.invoke.return_value = [
        {"repository_url": "https://api.github.com/repos/x/repo1",
         "title": "Fix pytest deprecation warnings", "html_url": "https://github.com/x/repo1/issues/1"},
        {"repository_url": "https://api.github.com/repos/y/repo2",
         "title": "Fix pytest deprecation warnings", "html_url": "https://github.com/y/repo2/issues/9"},  # same title, different repo -- now collapsed
        {"repository_url": "https://api.github.com/repos/z/repo3",
         "title": "Add type hints to core module", "html_url": "https://github.com/z/repo3/issues/2"},  # genuinely different title -- kept
    ]

    def _route(prompt_value):
        text = prompt_value.to_string()
        title = "Add type hints to core module" if "type hints" in text else "Fix pytest deprecation warnings"
        return ContributionMatch(
            repo_url="https://api.github.com/repos/x/repo1", issue_title=title,
            issue_url="https://github.com/x/repo1/issues/1", is_match=True, relevance_reason="ok",
        ).model_dump_json()

    fake_llm = RunnableLambda(_route)

    results = contribution_filter_chain.run_contribution_search(fake_llm, ["Python"], "Python")

    assert len(results) == 2  # the two "Fix pytest..." entries collapsed to one, the distinct title kept

