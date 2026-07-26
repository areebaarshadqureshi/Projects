from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from prompts.repo_audit_prompt import repo_audit_prompt
from schemas.audit_result import AuditResult
from utils.llm_output_cleaning import clean_llm_json_output


def build_repo_audit_chain(llm):
    parser = PydanticOutputParser(pydantic_object=AuditResult)
    chain = repo_audit_prompt.partial(
        format_instructions=parser.get_format_instructions()
    ) | llm | clean_llm_json_output | parser
    return chain


def run_repo_audit(llm, repo_data: dict) -> AuditResult:
    chain = build_repo_audit_chain(llm)
    return chain.invoke({
        "name": repo_data["name"],
        "description": repo_data["description"],
        "language": repo_data["language"],
        "topics": ", ".join(repo_data["topics"]),
        "commit_count": repo_data["commit_count"],
        "readme": repo_data["readme"][:3000],
    })


# --- v2: RunnableBranch for low-confidence clarifying-question routing ---

CLARIFYING_QUESTION_TEMPLATE = """This repo's documentation was flagged as unclear during an audit.

Repo: {repo_name}
Notes from the audit: {notes}

Ask ONE short, specific clarifying question that would help resolve the ambiguity.
Return only the question text, nothing else.
"""

clarifying_question_prompt = ChatPromptTemplate.from_template(CLARIFYING_QUESTION_TEMPLATE)


def _needs_clarification(audit_result: AuditResult) -> bool:
    return audit_result.confidence == "low"


def _add_clarifying_question(llm):
    question_chain = clarifying_question_prompt | llm | StrOutputParser()

    def _run(audit_result: AuditResult) -> AuditResult:
        question = question_chain.invoke({
            "repo_name": audit_result.repo_name,
            "notes": audit_result.notes,
        })
        return audit_result.model_copy(update={"clarifying_question": question.strip()})

    return RunnableLambda(_run)


def _skip_clarifying_question(audit_result: AuditResult) -> AuditResult:
    return audit_result.model_copy(update={"clarifying_question": ""})


def build_full_audit_chain(llm):
    """
    Per-repo audit chain designed for .batch() (Step 12: concurrent repo auditing).

    RunnableBranch routes each AuditResult after the base audit call:
      - confidence == "low"        -> second LLM call generates a clarifying_question
      - confidence == "high"/other -> clarifying_question set to "" (no second call)
    """
    base_chain = build_repo_audit_chain(llm)

    router = RunnableBranch(
        (_needs_clarification, _add_clarifying_question(llm)),
        RunnableLambda(_skip_clarifying_question),
    )

    def _prepare_input(repo_data: dict) -> dict:
        return {
            "name": repo_data["name"],
            "description": repo_data["description"],
            "language": repo_data["language"],
            "topics": ", ".join(repo_data["topics"]),
            "commit_count": repo_data["commit_count"],
            "readme": repo_data["readme"][:3000],
        }

    return RunnableLambda(_prepare_input) | base_chain | router
