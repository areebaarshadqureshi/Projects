"""
Tests for configs/llm_client.py's get_llm(). Did not exist before.

Only tests the branch-selection and missing-API-key error paths --
deliberately does NOT construct a real ChatGroq/ChatHuggingFace client
or make a real API call, since that needs a real key and costs money/quota.
"""

import pytest
import configs.llm_client as llm_client


def test_get_llm_raises_clear_error_when_groq_key_missing(monkeypatch):
    monkeypatch.setattr(llm_client, "ENVIRONMENT", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        llm_client.get_llm()


def test_get_llm_raises_clear_error_when_hf_key_missing(monkeypatch):
    monkeypatch.setattr(llm_client, "ENVIRONMENT", "production")
    monkeypatch.delenv("HF_API_TOKEN", raising=False)

    with pytest.raises(ValueError, match="HF_API_TOKEN"):
        llm_client.get_llm()


def test_get_llm_raises_clear_error_for_unknown_environment(monkeypatch):
    monkeypatch.setattr(llm_client, "ENVIRONMENT", "not_a_real_environment")

    with pytest.raises(ValueError, match="Unknown ENVIRONMENT"):
        llm_client.get_llm()
