from schemas.final_report import FinalReport
from schemas.audit_result import AuditResult
from schemas.gap_analysis_result import GapAnalysisResult, SuggestedProject
from schemas.contribution_match import ContributionMatch


def _fake_report():
    return FinalReport(
        username="octocat",
        overall_score=7.5,
        documentation_avg=7.5,
        summary="Solid progress overall.",
        strengths=["Clear README in repo-a"],
        weaknesses=["No tests in repo-b"],
        most_impressive_repo="repo-a (9/10 documentation, 8/10 structure)",
        best_repo_to_improve="repo-b (3/10 documentation, 4/10 structure)",
        top_missing_technologies=["Docker", "CI/CD"],
        recommended_learning_order=["Docker", "CI/CD"],
        roadmap_90_day=["Add tests to repo-b", "Learn Docker basics"],
        recruiter_readiness_pct=72,
        audit_results=[
            AuditResult(repo_name="repo-a", doc_quality_score=9, structure_score=8, confidence="high", notes="Good."),
        ],
        gap_analysis=GapAnalysisResult(
            missing_skills=["Docker"],
            suggested_projects=[SuggestedProject(
                title="Distributed Event Ticketing System",
                tech_stack=["Docker", "Redis", "RabbitMQ"],
                real_world_challenge="Handle concurrent ticket purchases without overselling",
            )],
        ),
        contributions=[ContributionMatch(
            repo_url="https://api.github.com/repos/x/y", issue_title="Fix bug",
            issue_url="https://github.com/x/y/issues/1", is_match=True, relevance_reason="Good fit",
        )],
    )


def test_to_markdown_includes_every_new_section():
    md = _fake_report().to_markdown()

    for expected in [
        "# GitHub Portfolio Report", "## Strengths", "## Weaknesses",
        "## Most Impressive Repository", "## Best Repository to Improve",
        "## Top Missing Technologies", "## Recommended Learning Order",
        "## Recommended Portfolio Projects", "## Recommended Open-Source Issues",
        "## 90-Day Roadmap", "Distributed Event Ticketing System",
        "Docker, Redis, RabbitMQ",  # tech_stack joined
        "Handle concurrent ticket purchases",
    ]:
        assert expected in md


def test_to_markdown_excludes_non_matching_contributions():
    report = _fake_report()
    report.contributions.append(ContributionMatch(
        repo_url="https://api.github.com/repos/a/b", issue_title="Irrelevant issue",
        issue_url="https://github.com/a/b/issues/2", is_match=False, relevance_reason="Not a fit",
    ))
    md = report.to_markdown()
    assert "Irrelevant issue" not in md
