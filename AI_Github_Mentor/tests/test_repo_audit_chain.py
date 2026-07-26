"""
Unit tests for chains/repo_audit_chain.py.

Mocks the LLM entirely (FakeListLLM) so these run fast, offline, and
without hitting any rate limits — exactly the point of this test file
per the doc's Step 16.
"""

from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda
from chains.repo_audit_chain import run_repo_audit, build_full_audit_chain
from schemas.audit_result import AuditResult


SAMPLE_REPO = {
    "name": "fake-repo",
    "description": "",
    "language": "Python",
    "topics": [],
    "commit_count": 1,
    "readme": "",
}


def test_run_repo_audit_returns_audit_result():
    fake_json = AuditResult(
        repo_name="fake-repo", doc_quality_score=5, structure_score=5,
        confidence="low", notes="test", clarifying_question="What does this do?",
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = run_repo_audit(fake_llm, SAMPLE_REPO)

    assert isinstance(result, AuditResult)
    assert result.repo_name == "fake-repo"
    assert result.confidence == "low"


def test_run_repo_audit_high_confidence_has_no_question_required():
    fake_json = AuditResult(
        repo_name="clear-repo", doc_quality_score=9, structure_score=8,
        confidence="high", notes="Well documented",
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = run_repo_audit(fake_llm, {**SAMPLE_REPO, "name": "clear-repo"})

    assert result.confidence == "high"
    assert result.clarifying_question == ""


def test_run_repo_audit_truncates_long_readmes():
    # readme[:3000] happens inside run_repo_audit before it ever reaches the
    # prompt -- confirm a long README doesn't crash the chain
    long_readme = "x" * 10000
    fake_json = AuditResult(
        repo_name="long-readme-repo", doc_quality_score=6, structure_score=6,
        confidence="high", notes="ok",
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = run_repo_audit(fake_llm, {**SAMPLE_REPO, "name": "long-readme-repo", "readme": long_readme})

    assert isinstance(result, AuditResult)


# --- v2: build_full_audit_chain (.batch() + RunnableBranch) ---

def _router_llm(repos_by_name: dict):
    """A fake LLM (as a RunnableLambda) that inspects the prompt text and
    returns the right canned response based on which repo/prompt it is --
    needed because .batch() may call things concurrently, so a simple
    FakeListLLM's fixed response order isn't reliable here."""
    def _route(prompt_value):
        text = prompt_value.to_string()
        if "flagged as unclear" in text:
            return "What does this repo actually do?"
        for name, audit_json in repos_by_name.items():
            if name in text:
                return audit_json
        raise ValueError("no matching fake response for prompt")
    return RunnableLambda(_route)


def test_build_full_audit_chain_batch_skips_second_call_for_high_confidence():
    high_json = AuditResult(repo_name="repo-high", doc_quality_score=8, structure_score=7,
                             confidence="high", notes="clear").model_dump_json()
    low_json = AuditResult(repo_name="repo-low", doc_quality_score=3, structure_score=2,
                            confidence="low", notes="unclear").model_dump_json()
    fake_llm = _router_llm({"repo-high": high_json, "repo-low": low_json})

    chain = build_full_audit_chain(fake_llm)
    repos = [
        {**SAMPLE_REPO, "name": "repo-high"},
        {**SAMPLE_REPO, "name": "repo-low"},
    ]
    results = chain.batch(repos)

    by_name = {r.repo_name: r for r in results}
    assert by_name["repo-high"].confidence == "high"
    assert by_name["repo-high"].clarifying_question == ""  # second call skipped
    assert by_name["repo-low"].confidence == "low"
    assert by_name["repo-low"].clarifying_question != ""  # second call made


def test_build_full_audit_chain_preserves_repo_identity_across_batch():
    json_a = AuditResult(repo_name="alpha", doc_quality_score=5, structure_score=5,
                          confidence="high", notes="a").model_dump_json()
    json_b = AuditResult(repo_name="beta", doc_quality_score=6, structure_score=6,
                          confidence="high", notes="b").model_dump_json()
    fake_llm = _router_llm({"alpha": json_a, "beta": json_b})

    chain = build_full_audit_chain(fake_llm)
    repos = [{**SAMPLE_REPO, "name": "alpha"}, {**SAMPLE_REPO, "name": "beta"}]
    results = chain.batch(repos)

    assert [r.repo_name for r in results] == ["alpha", "beta"]  # order preserved
