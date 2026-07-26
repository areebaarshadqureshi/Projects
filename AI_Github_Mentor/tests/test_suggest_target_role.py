"""
Tests for suggest_target_role -- moved into utils/skill_extractor.py
from app/streamlit_app.py in v2. Had no tests before.
"""

from utils.skill_extractor import suggest_target_role


def test_suggest_target_role_picks_highest_scoring_role():
    repos = [
        {"description": "A LangChain RAG pipeline using transformers and BERT for sentiment",
         "topics": ["nlp", "huggingface"], "language": "Python", "readme": ""},
    ]
    assert suggest_target_role(repos) == "NLP Engineer"


def test_suggest_target_role_returns_none_for_empty_signal():
    repos = [{"description": "", "topics": [], "language": "", "readme": ""}]
    assert suggest_target_role(repos) is None


def test_suggest_target_role_returns_none_for_no_repos():
    assert suggest_target_role([]) is None


def test_suggest_target_role_distinguishes_data_analyst_from_data_scientist():
    da_repos = [{"description": "Power BI dashboard with DAX measures and SQL reporting",
                 "topics": [], "language": "", "readme": ""}]
    ds_repos = [{"description": "Regression and classification models with feature engineering",
                 "topics": [], "language": "", "readme": ""}]
    assert suggest_target_role(da_repos) == "Data Analyst"
    assert suggest_target_role(ds_repos) == "Data Scientist"
