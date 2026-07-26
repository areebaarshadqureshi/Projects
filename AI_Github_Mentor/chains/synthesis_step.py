"""
chains/synthesis_step.py

Runs run_gap_analysis() and run_contribution_search() CONCURRENTLY via
RunnableParallel, since neither depends on the other's output -- both
only need `skills` (contribution search also needs a language).

v2: max_issues is now threaded through from the top of the pipeline,
same as target_role already was -- previously this always silently
used run_contribution_search's default of 10, with no way for a caller
to actually control it despite the tool itself supporting the param.
"""

from langchain_core.runnables import RunnableParallel, RunnableLambda
from chains.gap_analysis_chain import run_gap_analysis
from chains.contribution_filter_chain import run_contribution_search


def build_synthesis_step(llm, language: str, target_role: str | None = None, max_issues: int = 10):
    """
    Input: list[str] of skills.
    Output: {"gap_analysis": GapAnalysisResult, "contributions": list[ContributionMatch]}
    """
    return RunnableParallel(
        gap_analysis=RunnableLambda(lambda skills: run_gap_analysis(llm, skills, target_role)),
        contributions=RunnableLambda(
            lambda skills: run_contribution_search(llm, skills, language, max_issues=max_issues)
        ),
    )
