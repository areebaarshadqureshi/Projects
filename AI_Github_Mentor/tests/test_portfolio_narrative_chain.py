from langchain_core.language_models.fake import FakeListLLM
from chains.portfolio_narrative_chain import run_portfolio_narrative
from schemas.portfolio_narrative import PortfolioNarrative


def test_run_portfolio_narrative_returns_structured_result():
    fake_json = PortfolioNarrative(
        strengths=["Clear README in repo-a"],
        weaknesses=["No tests in repo-b"],
        recommended_learning_order=["Docker", "CI/CD"],
        roadmap_90_day=["Add tests to repo-b"] * 8,
        recruiter_readiness_pct=72,
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = run_portfolio_narrative(
        fake_llm,
        repo_summaries="- repo-a: documentation 9/10, structure 8/10, confidence high. Good docs.",
        missing_skills=["Docker", "CI/CD"],
        overall_score=7.5,
        target_role="Data Scientist",
    )

    assert isinstance(result, PortfolioNarrative)
    assert result.recruiter_readiness_pct == 72
    assert "Docker" in result.recommended_learning_order


def test_run_portfolio_narrative_handles_no_target_role():
    fake_json = PortfolioNarrative(
        strengths=["ok"], weaknesses=["ok"],
        recommended_learning_order=[], roadmap_90_day=["item"],
        recruiter_readiness_pct=50,
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = run_portfolio_narrative(
        fake_llm, repo_summaries="none", missing_skills=[], overall_score=5.0, target_role=None,
    )

    assert isinstance(result, PortfolioNarrative)
