"""
Unit tests for chains/gap_analysis_chain.py.

Mocks BOTH the embeddings model (no huggingface.co network call) and the
LLM (FakeListLLM), and builds a tiny real FAISS index in a temp
directory so the test exercises actual retrieval logic rather than just
mocking it away entirely.
"""

import hashlib
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

import chains.gap_analysis_chain as gap_analysis_chain
from schemas.gap_analysis_result import GapAnalysisResult, SuggestedProject


class FakeDeterministicEmbeddings(Embeddings):
    """Deterministic hash-based embeddings -- stands in for HuggingFaceEmbeddings
    so tests run offline without downloading a real model."""

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    """Builds a tiny real FAISS index in a temp dir and points the chain at it."""
    docs = [
        Document(page_content="Role: ML Engineer\nRequired skills: Docker, MLOps, Python", metadata={"role": "ML Engineer"}),
        Document(page_content="Role: Data Analyst\nRequired skills: SQL, Power BI, Excel", metadata={"role": "Data Analyst"}),
    ]
    embeddings = FakeDeterministicEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(tmp_path / "job_requirements"))

    monkeypatch.setattr(gap_analysis_chain, "VECTORSTORE_DIR", tmp_path)
    monkeypatch.setattr(gap_analysis_chain, "get_embeddings", lambda: FakeDeterministicEmbeddings())
    return tmp_path


def test_run_gap_analysis_returns_gap_analysis_result(fake_index):
    fake_json = GapAnalysisResult(
        missing_skills=["Docker", "MLOps"],
        suggested_projects=[SuggestedProject(title="Deploy a model with Docker", tech_stack=["Docker"], real_world_challenge="Package and serve a model reliably")],
    ).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = gap_analysis_chain.run_gap_analysis(fake_llm, ["Python", "Pandas"])

    assert isinstance(result, GapAnalysisResult)
    assert "Docker" in result.missing_skills


def test_run_gap_analysis_handles_empty_skill_list(fake_index):
    fake_json = GapAnalysisResult(missing_skills=[], suggested_projects=[]).model_dump_json()
    fake_llm = FakeListLLM(responses=[fake_json])

    result = gap_analysis_chain.run_gap_analysis(fake_llm, [])

    assert isinstance(result, GapAnalysisResult)


def test_retrieve_job_requirements_returns_docs_directly(fake_index):
    """
    Direct test of the extracted retrieval function, now that it's its
    own unit (added when it got pulled out for @traceable instrumentation).
    """
    from langchain_community.vectorstores import FAISS

    # Use the fixture's monkeypatched fake embeddings, not a fresh import
    # of the real one -- otherwise this would try to load sentence-transformers.
    embeddings = gap_analysis_chain.get_embeddings()
    vectorstore = FAISS.load_local(
        str(gap_analysis_chain.VECTORSTORE_DIR / "job_requirements"), embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = gap_analysis_chain._retrieve_job_requirements(vectorstore, "Python Pandas", None)
    assert len(docs) > 0


# --- previously untested: target_role filtering and its fallback path ---
#
# v1's test suite never passed target_role at all, so the filter={"role": ...}
# branch and the "filter matched nothing -> retry unfiltered" safety net in
# run_gap_analysis were never actually exercised by a test. Both fixed docs
# in fake_index carry distinct role metadata specifically so these two tests
# can prove real filtering behavior, not just that the code runs.

def test_run_gap_analysis_filters_retrieval_by_target_role(fake_index):
    """
    fake_index has two docs: one metadata={"role": "ML Engineer"} (mentions
    Docker/MLOps), one metadata={"role": "Data Analyst"} (mentions Power BI).
    A fake LLM that inspects the actual prompt text proves the filter
    genuinely restricted retrieval to the matching doc only.
    """
    def _route(prompt_value):
        text = prompt_value.to_string()
        saw_only_ml_engineer_doc = "MLOps" in text and "Power BI" not in text
        result = GapAnalysisResult(
            missing_skills=["filter_worked"] if saw_only_ml_engineer_doc else ["filter_failed"],
            suggested_projects=[],
        )
        return result.model_dump_json()

    fake_llm = RunnableLambda(_route)

    result = gap_analysis_chain.run_gap_analysis(fake_llm, ["Python"], target_role="ML Engineer")

    assert result.missing_skills == ["filter_worked"]


def test_run_gap_analysis_falls_back_to_unfiltered_when_role_matches_nothing(fake_index):
    """
    A target_role with no matching metadata in the index should NOT return
    an empty context to the LLM -- run_gap_analysis retries unfiltered.
    This proves both docs come back when the filter matches zero documents.
    """
    def _route(prompt_value):
        text = prompt_value.to_string()
        saw_both_docs = "MLOps" in text and "Power BI" in text
        result = GapAnalysisResult(
            missing_skills=["fallback_worked"] if saw_both_docs else ["fallback_failed"],
            suggested_projects=[],
        )
        return result.model_dump_json()

    fake_llm = RunnableLambda(_route)

    result = gap_analysis_chain.run_gap_analysis(fake_llm, ["Python"], target_role="Role That Does Not Exist")

    assert result.missing_skills == ["fallback_worked"]
