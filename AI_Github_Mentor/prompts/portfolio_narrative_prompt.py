from langchain_core.prompts import ChatPromptTemplate

PORTFOLIO_NARRATIVE_TEMPLATE = """Repo-by-repo audit results:
{repo_summaries}

Overall portfolio score: {overall_score}/10
Target role: {target_role}
Missing skills identified: {missing_skills}

Based on this, provide:
1. strengths: 2-4 concrete strengths, each citing specific evidence from the
   repos above (name the repo or the specific thing that's good) -- not
   generic praise like "good coding practices".
2. weaknesses: 2-4 concrete, actionable weaknesses -- name the specific gap
   and ideally which repo it's in.
3. recommended_learning_order: reorder the missing skills into a sensible
   learning sequence, foundational skills before advanced ones.
4. roadmap_90_day: 8-10 flat action items for the next 90 days -- concrete
   and sequenced (earlier items first), but not tied to specific
   days/weeks.
5. recruiter_readiness_pct: 0-100, weighing documentation quality, code
   structure, and how large the skill gaps are, together.

{format_instructions}
"""

portfolio_narrative_prompt = ChatPromptTemplate.from_template(PORTFOLIO_NARRATIVE_TEMPLATE)
