from langchain_core.output_parsers import PydanticOutputParser
from schemas.portfolio_narrative import PortfolioNarrative
from prompts.portfolio_narrative_prompt import portfolio_narrative_prompt
from utils.llm_output_cleaning import clean_llm_json_output


def build_portfolio_narrative_chain(llm):
    parser = PydanticOutputParser(pydantic_object=PortfolioNarrative)
    return (
        portfolio_narrative_prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | clean_llm_json_output
        | parser
    )


def run_portfolio_narrative(
    llm, repo_summaries: str, missing_skills: list[str],
    overall_score: float, target_role: str | None,
) -> PortfolioNarrative:
    chain = build_portfolio_narrative_chain(llm)
    return chain.invoke({
        "repo_summaries": repo_summaries,
        "overall_score": overall_score,
        "target_role": target_role or "Any / no preference",
        "missing_skills": ", ".join(missing_skills) if missing_skills else "none identified",
    })
