from langchain_core.language_models.fake import FakeListLLM
from chains.summary_chain import run_summary


def test_run_summary_invokes_prompt_with_all_fields():
    fake_llm = FakeListLLM(responses=["You're off to a solid start. Consider adding Docker next. Keep shipping."])

    result = run_summary(fake_llm, overall_score=7.5, repo_count=4, top_missing_skill="Docker")

    assert isinstance(result, str)
    assert len(result) > 0
