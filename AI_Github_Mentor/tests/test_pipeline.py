"""
Tests for core/pipeline.py.

This file didn't exist in v1 -- app/orchestrator.py had zero tests.
It matters beyond just coverage: this is specifically the test suite
that proves the backend/frontend boundary decided in Stage 1 actually
holds. Every dependency here is mocked at the core.pipeline import
site, and nothing in this file (or core/pipeline.py itself) imports
Streamlit -- if it ever did, that import would need to appear here too,
which is the tell that the boundary had been violated.
"""

import inspect
from unittest.mock import patch, MagicMock, ANY
import core.pipeline as pipeline
from schemas.final_report import FinalReport
from schemas.audit_result import AuditResult
from schemas.gap_analysis_result import GapAnalysisResult
from schemas.contribution_match import ContributionMatch
from schemas.portfolio_narrative import PortfolioNarrative


def test_core_pipeline_has_no_streamlit_dependency():
    """
    The whole point of the v2 backend/frontend split: this module should
    be safe to run with Streamlit not even installed. Checking actual
    import statements, not just absence of the word anywhere in the
    file -- the docstring here legitimately explains the design
    decision and mentions Streamlit by name.
    """
    import ast
    tree = ast.parse(inspect.getsource(pipeline))
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)
    assert not any("streamlit" in name.lower() for name in imported_names)


@patch("core.pipeline.build_full_audit_chain")
@patch("core.pipeline.github_repo_tool")
def test_run_audit_phase_passes_max_repos_through(mock_repo_tool, mock_build_chain):
    mock_repo_tool.invoke.return_value = [{"name": "repo1", "language": "Python"}]
    mock_chain = MagicMock()
    mock_chain.batch.return_value = ["fake_audit_result"]
    mock_build_chain.return_value = mock_chain

    repos_data, audit_results = pipeline.run_audit_phase(MagicMock(), "octocat", max_repos=15)

    mock_repo_tool.invoke.assert_called_once_with({"username": "octocat", "max_repos": 15})
    assert audit_results == ["fake_audit_result"]


def _fake_audit_result(repo_name, notes="original notes", doc_quality_score=8):
    return AuditResult(
        repo_name=repo_name, doc_quality_score=doc_quality_score,
        structure_score=7, confidence="high", notes=notes,
    )


def test_apply_clarification_answers_appends_only_when_answered():
    results = [_fake_audit_result("repo-a"), _fake_audit_result("repo-b")]

    pipeline._apply_clarification_answers(results, {"repo-a": "It's a personal fork, not original work."})

    assert "Developer clarification: It's a personal fork" in results[0].notes
    assert "Developer clarification" not in results[1].notes


def test_apply_clarification_answers_ignores_blank_answers():
    results = [_fake_audit_result("repo-a")]

    pipeline._apply_clarification_answers(results, {"repo-a": "   "})

    assert results[0].notes == "original notes"


def _fake_narrative():
    return PortfolioNarrative(
        strengths=["Clear README in repo-a"],
        weaknesses=["No tests in repo-b"],
        recommended_learning_order=["Docker", "CI/CD"],
        roadmap_90_day=["Add tests to repo-b", "Learn Docker basics"],
        recruiter_readiness_pct=72,
    )


@patch("core.pipeline.run_portfolio_narrative")
@patch("core.pipeline.run_summary")
@patch("core.pipeline.build_synthesis_step")
@patch("core.pipeline.extract_skills_from_repos")
def test_run_synthesis_phase_builds_final_report_with_summary(
    mock_extract_skills, mock_build_step, mock_run_summary, mock_narrative,
):
    mock_extract_skills.return_value = ["Python", "Pandas"]
    mock_step = MagicMock()
    mock_step.invoke.return_value = {
        "gap_analysis": GapAnalysisResult(missing_skills=["Docker"], suggested_projects=[]),
        "contributions": [],
    }
    mock_build_step.return_value = mock_step
    mock_run_summary.return_value = "You're making solid progress."
    mock_narrative.return_value = _fake_narrative()

    audit_results = [_fake_audit_result("repo-a", doc_quality_score=8), _fake_audit_result("repo-b", doc_quality_score=6)]
    repos_data = [{"name": "repo-a", "language": "Python"}]

    result = pipeline.run_synthesis_phase(
        MagicMock(), "octocat", repos_data, audit_results, clarification_answers={},
    )

    assert isinstance(result, FinalReport)
    assert result.overall_score == 7.0  # (8 + 6) / 2
    assert result.documentation_avg == 7.0
    assert result.summary == "You're making solid progress."
    assert result.strengths == ["Clear README in repo-a"]
    assert result.recruiter_readiness_pct == 72
    assert "repo-a" in result.most_impressive_repo  # higher doc_quality_score
    assert "repo-b" in result.best_repo_to_improve   # lower doc_quality_score
    mock_run_summary.assert_called_once_with(
        ANY, overall_score=7.0, repo_count=2, top_missing_skill="Docker",
    )


@patch("core.pipeline.run_portfolio_narrative")
@patch("core.pipeline.run_summary")
@patch("core.pipeline.build_synthesis_step")
@patch("core.pipeline.extract_skills_from_repos")
def test_run_synthesis_phase_passes_max_issues_through(
    mock_extract_skills, mock_build_step, mock_run_summary, mock_narrative,
):
    mock_extract_skills.return_value = ["Python"]
    mock_step = MagicMock()
    mock_step.invoke.return_value = {
        "gap_analysis": GapAnalysisResult(missing_skills=[], suggested_projects=[]),
        "contributions": [],
    }
    mock_build_step.return_value = mock_step
    mock_run_summary.return_value = "summary text"
    mock_narrative.return_value = _fake_narrative()

    pipeline.run_synthesis_phase(
        MagicMock(), "octocat", [{"name": "r", "language": "Python"}],
        [_fake_audit_result("r")], clarification_answers={}, max_issues=25,
    )

    _, kwargs = mock_build_step.call_args
    args = mock_build_step.call_args.args
    assert 25 in args or kwargs.get("max_issues") == 25


def test_pick_most_impressive_and_worst_repo_picks_by_doc_quality_score():
    results = [
        _fake_audit_result("weak-repo", doc_quality_score=3),
        _fake_audit_result("strong-repo", doc_quality_score=9),
    ]

    best, worst = pipeline._pick_most_impressive_and_worst_repo(results)

    assert "strong-repo" in best
    assert "weak-repo" in worst


def test_pick_most_impressive_and_worst_repo_handles_empty_list():
    best, worst = pipeline._pick_most_impressive_and_worst_repo([])
    assert best == "none audited"
    assert worst == "none audited"


def test_build_repo_summaries_text_includes_every_repo():
    results = [_fake_audit_result("repo-a"), _fake_audit_result("repo-b")]
    text = pipeline._build_repo_summaries_text(results)
    assert "repo-a" in text
    assert "repo-b" in text
