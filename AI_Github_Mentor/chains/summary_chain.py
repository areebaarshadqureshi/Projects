from langchain_core.output_parsers import StrOutputParser
from prompts.summary_prompt import summary_prompt


def run_summary(llm, overall_score: float, repo_count: int, top_missing_skill: str) -> str:
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({
        "overall_score": overall_score,
        "repo_count": repo_count,
        "top_missing_skill": top_missing_skill,
    })
