KNOWN_SKILLS = [
    "XGBoost", "LangChain", "FastAPI", "Pandas", "NumPy", "Scikit-learn",
    "PyTorch", "TensorFlow", "SHAP", "Streamlit", "Docker", "SQL",
    "Hugging Face", "Transformers", "spaCy", "NLTK", "OpenCV",
    # TODO: expand this vocabulary as needed
]


def extract_skills_from_repos(repo_data_list: list[dict]) -> list[str]:
    found = set()
    for repo in repo_data_list:
        text = f"{repo['description']} {repo['readme']} {' '.join(repo['topics'])} {repo['language']}".lower()
        for skill in KNOWN_SKILLS:
            if skill.lower() in text:
                found.add(skill)
    return sorted(found)


# Keywords/topics that show up on AI/ML/data-focused repos even when none of
# KNOWN_SKILLS matches (e.g. a repo that talks about "neural network" or
# "dataset" without naming a specific library).
AI_DOMAIN_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "data science", "artificial intelligence",
    " ai ", "llm", "large language model", "computer vision", "dataset",
    "model training", "data analysis", "regression", "classification",
    "clustering", "generative ai", "chatbot", "recommendation system",
]
AI_SIGNAL_LANGUAGES = {"jupyter notebook", "python", "r"}


def has_ai_domain_signal(repo_data_list: list[dict], detected_skills: list[str] | None = None) -> bool:
    """
    Soft heuristic, not a hard classifier: True if there's at least one
    plausible signal that this profile does AI/ML/data work. Used to
    surface a gentle warning rather than block analysis outright --
    a profile with no README topics filled in should not get hard-rejected
    on a false negative.
    """
    if detected_skills:
        return True
    for repo in repo_data_list:
        language = (repo.get("language") or "").lower()
        text = f"{repo.get('description', '')} {repo.get('readme', '')} {' '.join(repo.get('topics', []))}".lower()
        if language in AI_SIGNAL_LANGUAGES and any(k in text for k in AI_DOMAIN_KEYWORDS):
            return True
        if any(k in text for k in AI_DOMAIN_KEYWORDS):
            return True
    return False


# Moved out of app/streamlit_app.py in v2 -- this is a keyword-matching
# heuristic over data the audit already returns (no extra LLM call, no
# pipeline changes), used to pre-select "Target role" in the UI if the
# user hasn't picked one. It's domain logic reasoning about repo content,
# not UI rendering, so it belongs here next to the other skill/domain
# heuristics rather than inside the Streamlit file.
ROLE_KEYWORDS = {
    "NLP Engineer": ["nlp", "transformer", "bert", "llm", "langchain", "huggingface",
                     "sentiment", "tokeniz", "roman urdu", "text classification"],
    "Machine Learning Engineer": ["mlops", "docker", "fastapi", "deployment", "pipeline",
                                  "xgboost", "pytorch", "tensorflow", "model serving", "ci/cd"],
    "Data Scientist": ["regression", "classification", "clustering", "eda", "scikit",
                       "forecast", "statistics", "feature engineering"],
    "Data Analyst": ["sql", "power bi", "powerbi", "tableau", "excel", "dax",
                     "reporting", "dashboard", "visualization"],
}


def suggest_target_role(repos_data: list[dict]) -> str | None:
    skills = extract_skills_from_repos(repos_data) or []
    blob = (" ".join(skills) + " " + " ".join(
        f"{r.get('description') or ''} {' '.join(r.get('topics') or [])}" for r in repos_data
    )).lower()
    if not blob.strip():
        return None
    scores = {role: sum(1 for kw in kws if kw in blob) for role, kws in ROLE_KEYWORDS.items()}
    best_role, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_role if best_score > 0 else None
