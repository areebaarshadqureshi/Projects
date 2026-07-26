"""
Tests for scripts/build_job_index.py. Did not exist in v1.

Only tests load_job_documents() directly against a temp directory of
JSON files -- deliberately does NOT call build_index() itself, since
that would require a real or fake embeddings model and a real FAISS
write. That combination is already covered indirectly by
test_gap_analysis_chain.py's fake_index fixture, which builds a FAISS
index the same way. Testing load_job_documents() in isolation is enough
to catch the actual risk here: a malformed or missing JSON field
silently producing garbage document text.
"""

import json
import scripts.build_job_index as build_job_index


def test_load_job_documents_parses_json_into_documents(tmp_path, monkeypatch):
    job_dir = tmp_path / "job_requirements"
    job_dir.mkdir()
    (job_dir / "data_scientist_karachi.json").write_text(json.dumps([
        {"role": "Data Scientist", "seniority": "entry-level",
         "required_skills": ["Python", "SQL"], "nice_to_have": ["Docker"]},
    ]))
    monkeypatch.setattr(build_job_index, "DATA_DIR", tmp_path)

    docs = build_job_index.load_job_documents()

    assert len(docs) == 1
    assert docs[0].metadata["role"] == "Data Scientist"
    assert "Python" in docs[0].page_content
    assert "Docker" in docs[0].page_content


def test_load_job_documents_handles_missing_nice_to_have(tmp_path, monkeypatch):
    """
    nice_to_have is read with .get(..., []) in the source -- proves that
    actually works rather than assuming it does, since a posting scraped
    with only required_skills is a realistic case.
    """
    job_dir = tmp_path / "job_requirements"
    job_dir.mkdir()
    (job_dir / "ml_engineer_karachi.json").write_text(json.dumps([
        {"role": "ML Engineer", "seniority": "mid", "required_skills": ["PyTorch"]},
    ]))
    monkeypatch.setattr(build_job_index, "DATA_DIR", tmp_path)

    docs = build_job_index.load_job_documents()

    assert len(docs) == 1
    assert "PyTorch" in docs[0].page_content


def test_load_job_documents_combines_multiple_files(tmp_path, monkeypatch):
    job_dir = tmp_path / "job_requirements"
    job_dir.mkdir()
    (job_dir / "role_a.json").write_text(json.dumps([{"role": "Role A", "seniority": "mid", "required_skills": []}]))
    (job_dir / "role_b.json").write_text(json.dumps([{"role": "Role B", "seniority": "mid", "required_skills": []}]))
    monkeypatch.setattr(build_job_index, "DATA_DIR", tmp_path)

    docs = build_job_index.load_job_documents()

    assert {d.metadata["role"] for d in docs} == {"Role A", "Role B"}
