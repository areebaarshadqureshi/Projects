"""
core/pipeline.py

The single seam between backend and frontend (see Stage 1 planning:
this file has zero Streamlit imports and zero UI logic -- it must be
fully testable and runnable from a plain script or pytest). Keeps
run_audit_phase() / run_synthesis_phase() as the only two functions
any UI (Streamlit, Colab notebook, or a test) needs to call.

v3 report redesign: FinalReport now includes strengths, weaknesses,
most_impressive_repo, best_repo_to_improve, top_missing_technologies,
recommended_learning_order, roadmap_90_day, and recruiter_readiness_pct.

Two different strategies used here on purpose:
  - most_impressive_repo / best_repo_to_improve / documentation_avg /
    top_missing_technologies are all computed DETERMINISTICALLY from
    audit_results and gap_analysis -- no LLM call, no hallucination
    risk, and free.
  - strengths / weaknesses / recommended_learning_order / roadmap_90_day
    / recruiter_readiness_pct genuinely need LLM judgment (qualitative
    synthesis across repos), so these come from the new
    chains/portfolio_narrative_chain.py.
"""

from schemas.final_report import FinalReport
from tools.github_repo_tool import github_repo_tool
from chains.repo_audit_chain import build_full_audit_chain
from chains.synthesis_step import build_synthesis_step
from chains.summary_chain import run_summary
from chains.portfolio_narrative_chain import run_portfolio_narrative
from utils.skill_extractor import extract_skills_from_repos


def run_audit_phase(llm, username: str, max_repos: int = 20):
    repos_data = github_repo_tool.invoke({"username": username, "max_repos": max_repos})
    chain = build_full_audit_chain(llm)
    audit_results = chain.batch(repos_data, config={"max_concurrency": 3})
    return repos_data, audit_results


def _apply_clarification_answers(audit_results: list, clarification_answers: dict) -> None:
    """
    Folds each answered clarifying question back into that repo's notes,
    in place. Repos with no answer (clarification_answers doesn't have
    an entry, or the value is empty/whitespace) are left untouched.
    """
    for result in audit_results:
        answer = clarification_answers.get(result.repo_name)
        if answer and answer.strip():
            result.notes = (
                f"{result.notes}\nDeveloper clarification: {answer.strip()}"
            )


def _pick_most_impressive_and_worst_repo(audit_results: list) -> tuple[str, str]:
    """
    Deterministic, not LLM-judged -- picks by doc_quality_score (ties
    broken by structure_score). Avoids asking the LLM to re-derive a
    judgment the audit phase already made numerically.
    """
    if not audit_results:
        return "none audited", "none audited"
    ranked = sorted(audit_results, key=lambda r: (r.doc_quality_score, r.structure_score))
    worst = ranked[0]
    best = ranked[-1]
    best_str = f"{best.repo_name} ({best.doc_quality_score}/10 documentation, {best.structure_score}/10 structure)"
    worst_str = f"{worst.repo_name} ({worst.doc_quality_score}/10 documentation, {worst.structure_score}/10 structure)"
    return best_str, worst_str


def _build_repo_summaries_text(audit_results: list) -> str:
    lines = []
    for repo in audit_results:
        lines.append(
            f"- {repo.repo_name}: documentation {repo.doc_quality_score}/10, "
            f"structure {repo.structure_score}/10, confidence {repo.confidence}. {repo.notes}"
        )
    return "\n".join(lines) if lines else "No repositories audited."


def run_synthesis_phase(llm, username: str, repos_data: list[dict],
                          audit_results: list, clarification_answers: dict,
                          target_role: str | None = None, max_issues: int = 10) -> FinalReport:
    _apply_clarification_answers(audit_results, clarification_answers)

    skills = extract_skills_from_repos(repos_data)
    primary_language = repos_data[0]["language"] if repos_data else "Python"

    step = build_synthesis_step(llm, primary_language, target_role, max_issues=max_issues)
    result = step.invoke(skills)

    doc_scores = [r.doc_quality_score for r in audit_results] or [0]
    overall_score = round(sum(doc_scores) / len(doc_scores), 1)
    documentation_avg = round(sum(doc_scores) / len(doc_scores), 1)

    missing_skills = result["gap_analysis"].missing_skills
    top_missing_skill = missing_skills[0] if missing_skills else "none identified"
    top_missing_technologies = missing_skills[:5]

    summary = run_summary(
        llm,
        overall_score=overall_score,
        repo_count=len(audit_results),
        top_missing_skill=top_missing_skill,
    )

    most_impressive_repo, best_repo_to_improve = _pick_most_impressive_and_worst_repo(audit_results)

    narrative = run_portfolio_narrative(
        llm,
        repo_summaries=_build_repo_summaries_text(audit_results),
        missing_skills=missing_skills,
        overall_score=overall_score,
        target_role=target_role,
    )

    return FinalReport(
        username=username,
        overall_score=overall_score,
        documentation_avg=documentation_avg,
        summary=summary,
        strengths=narrative.strengths,
        weaknesses=narrative.weaknesses,
        most_impressive_repo=most_impressive_repo,
        best_repo_to_improve=best_repo_to_improve,
        top_missing_technologies=top_missing_technologies,
        recommended_learning_order=narrative.recommended_learning_order,
        roadmap_90_day=narrative.roadmap_90_day,
        recruiter_readiness_pct=narrative.recruiter_readiness_pct,
        audit_results=audit_results,
        gap_analysis=result["gap_analysis"],
        contributions=result["contributions"],
    )
