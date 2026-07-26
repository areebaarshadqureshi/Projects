"""
Unit tests for chains/synthesis_step.py (RunnableParallel).

Same mocking strategy as test_gap_analysis_chain.py: fake embeddings +
a temp FAISS index so no real network/model download is needed.

v2 additions vs v1's version of this test: ContributionMatch now
requires is_match (the fake router response was updated accordingly),
and a new test proves max_issues actually reaches the tool call instead
of silently sticking at the default of 10.
"""

import hashlib
from unittest.mock import patch
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda

import chains.gap_analysis_chain as gap_analysis_chain
from chains.synthesis_step import build_synthesis_step
from schemas.gap_analysis_result import GapAnalysisResult, SuggestedProject
from schemas.contribution_match import ContributionMatch


class FakeDeterministicEmbeddings(Embeddings):
    def _vec(self, text):
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    docs = [Document(page_content="Role: ML Engineer\nRequired skills: Docker, MLOps", metadata={"role": "ML Engineer"})]
    FAISS.from_documents(docs, FakeDeterministicEmbeddings()).save_local(str(tmp_path / "job_requirements"))
    monkeypatch.setattr(gap_analysis_chain, "VECTORSTORE_DIR", tmp_path)
    monkeypatch.setattr(gap_analysis_chain, "get_embeddings", lambda: FakeDeterministicEmbeddings())
    return tmp_path


def _router_llm():
    def _route(prompt_value):
        text = prompt_value.to_string()
        if "Candidate GitHub issue" in text:
            return ContributionMatch(
                repo_url="https://api.github.com/repos/x/y", issue_title="t",
                issue_url="https://github.com/x/y/issues/1",
                is_match=True, relevance_reason="matches",
            ).model_dump_json()
        return GapAnalysisResult(
            missing_skills=["Docker"],
            suggested_projects=[SuggestedProject(title="p", tech_stack=["Docker"], real_world_challenge="r")],
        ).model_dump_json()
    return RunnableLambda(_route)


def test_synthesis_step_returns_both_keys(fake_index):
    with patch("tools.github_issue_tool.github_issue_tool") as mock_tool:
        mock_tool.invoke.return_value = [
            {"repository_url": "https://api.github.com/repos/x/y",
             "title": "t", "html_url": "https://github.com/x/y/issues/1"},
        ]
        step = build_synthesis_step(_router_llm(), "Python")
        result = step.invoke(["Python", "Pandas"])

    assert "gap_analysis" in result
    assert "contributions" in result
    assert isinstance(result["gap_analysis"], GapAnalysisResult)
    assert all(isinstance(c, ContributionMatch) for c in result["contributions"])


def test_synthesis_step_passes_max_issues_through_to_tool(fake_index):
    """
    Proves max_issues reaches the actual github_issue_tool.invoke() call
    -- previously this was always stuck at the tool's own default of 10
    with no way for a caller of build_synthesis_step to override it.
    """
    with patch("tools.github_issue_tool.github_issue_tool") as mock_tool:
        mock_tool.invoke.return_value = []
        step = build_synthesis_step(_router_llm(), "Python", max_issues=3)
        step.invoke(["Python"])

        mock_tool.invoke.assert_called_once_with({"language": "Python", "max_issues": 3})
