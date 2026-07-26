"""
chains/gap_analysis_chain.py

RAG step: retrieves relevant job requirement documents from the FAISS
index and generates the missing-skills / suggested-projects analysis.

v2 changes:
  - Uses the shared embeddings/embedding_config.get_embeddings() instead
    of a second, locally-defined get_embeddings() -- keeps build-time and
    query-time embeddings guaranteed identical (see embedding_config.py's
    own docstring for why that matters).
  - Imports the prompt from prompts/gap_analysis_prompt.py instead of
    inlining a duplicate copy of the template.
  - Accepts an optional target_role. If given, retrieval is filtered to
    that role's postings via FAISS metadata filtering (each Document was
    indexed with metadata={"role": ...} in scripts/build_job_index.py).
    Falls back to an unfiltered search if the filter returns nothing, so
    a role-name mismatch degrades gracefully instead of feeding the LLM
    an empty context.
"""

from langsmith import traceable
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from schemas.gap_analysis_result import GapAnalysisResult
from configs.settings import VECTORSTORE_DIR, TOP_K_RETRIEVAL
from embeddings.embedding_config import get_embeddings
from prompts.gap_analysis_prompt import gap_analysis_prompt as gap_prompt
from utils.llm_output_cleaning import clean_llm_json_output


@traceable(name="gap_analysis_retrieval")
def _retrieve_job_requirements(vectorstore, query: str, target_role: str | None):
    """
    The actual RAG retrieval step, pulled out of run_gap_analysis into its
    own function specifically so it gets its own named span in LangSmith.
    vectorstore.similarity_search() is a plain method call, not an LCEL
    Runnable -- LangSmith only auto-traces Runnables (chains built with
    `|`, RunnableParallel, etc.), so without this @traceable wrapper this
    step would be invisible in a trace even though the LLM call right
    after it (prompt | llm | parser) shows up automatically.
    """
    if target_role:
        retrieved_docs = vectorstore.similarity_search(
            query, k=TOP_K_RETRIEVAL, filter={"role": target_role}
        )
        if not retrieved_docs:
            # Role filter matched nothing -- retry unfiltered rather than
            # returning an empty context to the LLM.
            retrieved_docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    else:
        retrieved_docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    return retrieved_docs


def run_gap_analysis(llm, skills: list[str], target_role: str | None = None) -> GapAnalysisResult:
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR / "job_requirements"), embeddings,
        allow_dangerous_deserialization=True,
    )
    query = " ".join(skills)

    retrieved_docs = _retrieve_job_requirements(vectorstore, query, target_role)
    retrieved_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    parser = PydanticOutputParser(pydantic_object=GapAnalysisResult)
    chain = gap_prompt.partial(
        format_instructions=parser.get_format_instructions()
    ) | llm | clean_llm_json_output | parser

    return chain.invoke({
        "skills": ", ".join(skills),
        "retrieved_requirements": retrieved_text,
        "target_role": target_role or "any relevant role",
    })
