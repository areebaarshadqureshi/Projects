"""
app/streamlit_app.py -- the deployed entry point.

Contains NO pipeline logic. Owns exactly one thing:
st.session_state["pipeline_state"] -- and calls core.pipeline to do the
actual work. See core/pipeline.py's module docstring for the full
rationale on this split.

Three-phase flow, driven by pipeline_state["phase"]:
  "input"   -> username + target role form, "Analyze" button
  "audited" -> per-repo audit results, clarifying-question inputs,
               "Generate Full Report" button
  "done"    -> final report, download button, "Start Over"


Run locally with: streamlit run app/streamlit_app.py
Requires: pip install streamlit plotly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from configs.llm_client import get_llm
from core.pipeline import run_audit_phase, run_synthesis_phase
from utils.github_api_client import (
    get_user_profile,
    normalize_github_username,
    GitHubAPIError,
    looks_like_url,
    is_valid_github_profile_url,
    check_username_exists,
)
from utils.skill_extractor import extract_skills_from_repos, suggest_target_role, has_ai_domain_signal

st.set_page_config(
    page_title="AI GitHub Mentor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET_ROLES = {
    "Any / no preference": None,
    "Data Analyst": "Data Analyst",
    "Data Scientist": "Data Scientist",
    "Machine Learning Engineer": "Machine Learning Engineer",
    "NLP Engineer": "NLP Engineer",
}

# Fixed, not user-facing sliders -- a deliberate simplicity choice for
# the input form. core.pipeline supports overriding both if that ever
# changes.
MAX_REPOS_DEFAULT = 20
MAX_ISSUES_DEFAULT = 10

# --- Palette (single source of truth -- Python side, mirrored in CSS below) ---
COLOR_BG = "#08090B"
COLOR_BG_SUBTLE = "#0E1013"
COLOR_CARD = "#131519"
COLOR_CARD_HOVER = "#171A1F"
COLOR_PRIMARY = "#3ECF8E"
COLOR_PRIMARY_HOVER = "#34B87C"
COLOR_PRIMARY_SOFT = "rgba(62, 207, 142, 0.12)"
COLOR_TEXT = "#F4F5F7"
COLOR_TEXT_SECONDARY = "#9AA1AC"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_SUCCESS = "#3ECF8E"
COLOR_WARNING = "#E8B75C"
COLOR_DANGER = "#E5685F"
COLOR_BORDER = "#1F2228"
COLOR_BORDER_SOFT = "#191B20"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLOR_TEXT, family="Inter, -apple-system, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)
PLOTLY_CONFIG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Presentation-only helpers (charts, CSS, badges) -- no pipeline calls,
# no business logic, safe to leave in this file per the Stage 1 boundary.
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}

        .stApp {{
            background:
                radial-gradient(ellipse 1200px 500px at 50% -10%, rgba(62, 207, 142, 0.06), transparent),
                {COLOR_BG};
        }}
        .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}
        #MainMenu, footer, header {{ visibility: hidden; }}

        h1, h2, h3 {{ font-family: 'Manrope', 'Inter', sans-serif; letter-spacing: -0.01em; }}
        h1 {{ font-size: 2.5rem; font-weight: 800; color: {COLOR_TEXT}; }}
        h2 {{ font-size: 1.65rem; font-weight: 700; color: {COLOR_TEXT}; }}
        h3 {{ font-size: 1.3rem; font-weight: 700; color: {COLOR_TEXT}; }}
        p, .stMarkdown, label, .stCaption {{ color: {COLOR_TEXT_SECONDARY}; font-size: 1.02rem; }}
        div[data-testid="stMarkdownContainer"] p {{ font-size: 1.02rem; line-height: 1.6; }}
        hr {{ border-color: {COLOR_BORDER}; }}

        /* Bordered containers (st.container(border=True)) -> elegant cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
            padding: 0.35rem;
            transition: border-color 0.15s ease;
        }}

        /* st.metric -> stat card */
        div[data-testid="stMetric"] {{
            background: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
            padding: 20px 22px;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: 'Manrope', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            color: {COLOR_TEXT};
        }}
        div[data-testid="stMetricLabel"] {{
            color: {COLOR_TEXT_SECONDARY};
            font-weight: 500;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Buttons -- refined, no icons, subtle lift on hover.
           Primary buttons sit on a bright emerald fill, so black text reads
           with far better contrast than white on a light saturated color;
           secondary/outline buttons sit on a dark background, so they stay
           white/light. Streamlit nests the label in an inner <p>/<div> that
           doesn't reliably inherit color from the button element, so the
           color is forced with !important on the button AND its children. */
        .stButton > button, .stDownloadButton > button,
        div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {{
            background: {COLOR_PRIMARY};
            color: #08110D !important;
            border: none;
            border-radius: 10px;
            height: 46px;
            font-weight: 700;
            font-family: 'Manrope', sans-serif;
            letter-spacing: 0.01em;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
            transition: background 0.15s ease, transform 0.15s ease;
        }}
        .stButton > button *, .stDownloadButton > button *,
        div[data-testid="stButton"] button *, div[data-testid="stDownloadButton"] button * {{
            color: #08110D !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: {COLOR_PRIMARY_HOVER};
            color: #08110D !important;
            transform: translateY(-1px);
        }}
        .stButton > button:hover *, .stDownloadButton > button:hover * {{
            color: #08110D !important;
        }}
        .stButton > button:active, .stDownloadButton > button:active {{ transform: translateY(0); }}

        /* Secondary buttons (Start over, etc. -- not type="primary") */
        .stButton > button[kind="secondary"] {{
            background: transparent;
            color: {COLOR_TEXT} !important;
            border: 1px solid {COLOR_BORDER};
        }}
        .stButton > button[kind="secondary"] * {{ color: {COLOR_TEXT} !important; }}
        .stButton > button[kind="secondary"]:hover {{
            background: {COLOR_CARD_HOVER};
            border-color: {COLOR_TEXT_MUTED};
            color: {COLOR_TEXT} !important;
        }}
        .stButton > button[kind="secondary"]:hover * {{ color: {COLOR_TEXT} !important; }}

        /* Text inputs / selects -- polished fields */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
            background: {COLOR_BG_SUBTLE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            color: {COLOR_TEXT};
        }}
        .stTextInput input:focus {{
            border-color: {COLOR_PRIMARY};
            box-shadow: 0 0 0 1px {COLOR_PRIMARY};
        }}

        /* Progress bars */
        .stProgress > div > div {{ background: {COLOR_BORDER}; border-radius: 999px; }}
        .stProgress > div > div > div {{ background: {COLOR_PRIMARY}; border-radius: 999px; }}

        /* Tabs -- clean underline style, no icons */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            border-bottom: 1px solid {COLOR_BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {COLOR_TEXT_SECONDARY};
            font-weight: 600;
            font-size: 0.92rem;
            padding: 10px 4px;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLOR_TEXT} !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {COLOR_PRIMARY}; }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {COLOR_BG_SUBTLE};
            border-right: 1px solid {COLOR_BORDER};
        }}

        /* Hero -- restrained, enterprise-grade, no gradients-on-gradients */
        .hero {{
            padding: 0.2rem 0 1.6rem 0;
            margin-bottom: 0.4rem;
            border-bottom: 1px solid {COLOR_BORDER};
        }}
        .hero .eyebrow {{
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {COLOR_PRIMARY};
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .hero h1 {{ margin-bottom: 0.35rem; }}
        .hero p {{ color: {COLOR_TEXT_SECONDARY}; font-size: 1.15rem; margin-bottom: 0; max-width: 620px; }}

        /* Profile header card */
        .profile-card {{ display: flex; align-items: center; gap: 18px; flex-wrap: wrap; padding: 0.6rem; }}
        .profile-card img {{
            width: 60px; height: 60px; border-radius: 50%;
            border: 1px solid {COLOR_BORDER};
        }}
        .profile-name {{ font-family: 'Manrope', sans-serif; font-size: 1.35rem; font-weight: 700; color: {COLOR_TEXT}; }}
        .profile-handle {{ color: {COLOR_TEXT_MUTED}; font-size: 1rem; }}
        .profile-bio {{ color: {COLOR_TEXT_SECONDARY}; font-size: 1rem; margin-top: 4px; }}
        .profile-stats span {{
            color: {COLOR_TEXT_SECONDARY}; font-size: 0.85rem; margin-right: 18px;
        }}

        .section-label {{
            font-size: 0.88rem;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {COLOR_TEXT_MUTED};
            font-weight: 700;
            margin: 1.4rem 0 0.6rem 0;
        }}

        .repo-title {{ font-family: 'Manrope', sans-serif; font-size: 1.15rem; font-weight: 700; color: {COLOR_TEXT}; }}
        .badge {{
            display: inline-block;
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge-green {{ background: rgba(62, 207, 142, 0.12); color: {COLOR_SUCCESS}; }}
        .badge-yellow {{ background: rgba(232, 183, 92, 0.12); color: {COLOR_WARNING}; }}
        .badge-red {{ background: rgba(229, 104, 95, 0.12); color: {COLOR_DANGER}; }}

        .ai-summary {{
            background: {COLOR_PRIMARY_SOFT};
            border: 1px solid rgba(62, 207, 142, 0.25);
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
        }}
        .ai-summary h3 {{ font-size: 1.15rem; letter-spacing: 0.02em; margin-top: 0; margin-bottom: 0.8rem; }}
        .ai-summary-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.4rem 2rem;
        }}
        .ai-summary p {{ margin: 0 0 0.5rem 0; color: {COLOR_TEXT}; font-size: 1rem; }}
        .ai-summary-full {{ margin-top: 0.4rem; color: {COLOR_TEXT}; font-size: 1rem; }}
        @media (max-width: 768px) {{
            .ai-summary-grid {{ grid-template-columns: 1fr; }}
        }}

        /* Responsive tuning */
        @media (max-width: 768px) {{
            .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
            h1 {{ font-size: 1.5rem; }}
            .hero p {{ max-width: 100%; }}
            .profile-card {{ gap: 12px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_popup(message: str, kind: str = "info") -> None:
    """
    Custom auto-dismissing popup used in place of st.toast so it matches the
    icon-free dark theme. Pure CSS animation (fade in -> hold -> fade out,
    ~4s total) -- no JS timer, no user action needed to dismiss it. Each call
    gets a unique animation name via uuid so multiple popups in one run
    don't clash.
    """
    uid = uuid.uuid4().hex[:8]
    accent = {
        "info": COLOR_PRIMARY,
        "success": COLOR_SUCCESS,
        "warning": COLOR_WARNING,
        "error": COLOR_DANGER,
    }.get(kind, COLOR_PRIMARY)
    st.markdown(
        f"""
        <style>
        @keyframes popup-fade-{uid} {{
            0%   {{ opacity: 0; transform: translateY(10px); }}
            8%   {{ opacity: 1; transform: translateY(0); }}
            85%  {{ opacity: 1; transform: translateY(0); }}
            100% {{ opacity: 0; transform: translateY(10px); }}
        }}
        #popup-{uid} {{
            position: fixed;
            bottom: 28px;
            right: 28px;
            z-index: 9999;
            max-width: 360px;
            background: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-left: 3px solid {accent};
            border-radius: 10px;
            padding: 14px 18px;
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            font-weight: 500;
            line-height: 1.4;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.5);
            animation: popup-fade-{uid} 4s ease forwards;
            pointer-events: none;
        }}
        </style>
        <div id="popup-{uid}">{message}</div>
        """,
        unsafe_allow_html=True,
    )



def render_bar(label: str, pct: float, width: int = 10) -> str:
    """
    Text-based progress bar, e.g. "████████░░ 82%". Pure presentation --
    takes an already-computed percentage, makes no decisions about what
    that percentage means (that's core.pipeline's job).
    """
    pct = max(0, min(100, pct))
    filled = round((pct / 100) * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"{label}\n{bar} {round(pct)}%"


def score_badge(score: float) -> str:
    if score >= 8:
        cls, label = "badge-green", "Strong"
    elif score >= 5:
        cls, label = "badge-yellow", "Okay"
    else:
        cls, label = "badge-red", "Weak"
    return f'<span class="badge {cls}">{score}/10 &middot; {label}</span>'


def confidence_badge(confidence: str) -> str:
    if confidence == "high":
        return '<span class="badge badge-green">High confidence</span>'
    return '<span class="badge badge-yellow">Needs clarification</span>'


def score_tier_label(score: float) -> str:
    if score >= 8:
        return "Excellent"
    if score >= 6:
        return "Good"
    if score >= 4:
        return "Needs work"
    return "Weak"


# ---------------------------------------------------------------------------
# Input validation (Phase 3)
# ---------------------------------------------------------------------------

# Broader AI-signal vocabulary than skill_extractor.KNOWN_SKILLS -- this is
# deliberately looser (covers topics/README wording like "machine-learning"
# or "dataset") since its only job is a yes/no "is this an AI-ish profile"
# check, not skill scoring.
AI_DOMAIN_KEYWORDS = [
    "machine learning", "machine-learning", "deep learning", "deep-learning",
    "artificial intelligence", "neural network", "nlp", "llm", "genai",
    "computer vision", "data science", "dataset", "langchain", "transformer",
    "regression", "classification", "clustering", "chatbot", "rag",
    "scikit", "tensorflow", "pytorch", "huggingface", "hugging face",
]










# Role suggestion is a frontend-only heuristic, same pattern as
# has_ai_signal() above -- keyword matching over fields the audit already
# returns, no extra LLM call and no orchestrator/backend changes. It's used
# to pre-select "Target role" after analysis if the user hasn't picked one
# themselves; they can always override it.





def language_breakdown(repos_data: list[dict]) -> Counter:
    return Counter(r["language"] for r in repos_data if r.get("language"))


def skill_radar_chart(audit_results: list) -> go.Figure:
    """
    Radar axes are limited to fields the backend actually returns
    (doc_quality_score, structure_score, confidence) -- no invented
    categories like "Testing" or "Deployment" that were never scored.
    """
    n = len(audit_results) or 1
    doc_avg = sum(r.doc_quality_score for r in audit_results) / n
    structure_avg = sum(r.structure_score for r in audit_results) / n
    confidence_pct = (sum(1 for r in audit_results if r.confidence == "high") / n) * 10

    categories = ["Documentation", "Structure", "Confidence"]
    values = [round(doc_avg, 1), round(structure_avg, 1), round(confidence_pct, 1)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color=COLOR_PRIMARY, width=2),
            fillcolor="rgba(62, 207, 142, 0.20)",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 10], gridcolor=COLOR_BORDER,
                tickfont=dict(size=11, color=COLOR_TEXT_SECONDARY),
            ),
            angularaxis=dict(
                gridcolor=COLOR_BORDER,
                tickfont=dict(size=13, color=COLOR_TEXT),
            ),
        ),
        showlegend=False,
        title=dict(text="Skill Radar", font=dict(color=COLOR_TEXT, size=16, family="Manrope, sans-serif")),
        height=340,
        **PLOTLY_LAYOUT,
    )
    return fig


def repo_score_bar_chart(audit_results: list) -> go.Figure:
    """
    New chart (per-repo view, complements the portfolio-wide radar above):
    a grouped bar comparing documentation vs. structure score for every
    audited repo, so weak repos are identifiable by name at a glance.
    Same underlying fields as the radar (doc_quality_score,
    structure_score) -- no new backend data required.
    """
    names = [r.repo_name for r in audit_results]
    doc_scores = [r.doc_quality_score for r in audit_results]
    structure_scores = [r.structure_score for r in audit_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Documentation", x=names, y=doc_scores, marker_color=COLOR_PRIMARY))
    fig.add_trace(go.Bar(name="Structure", x=names, y=structure_scores, marker_color=COLOR_WARNING))
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Documentation vs. Structure by Repository",
            font=dict(color=COLOR_TEXT, size=16, family="Manrope, sans-serif"),
        ),
        yaxis=dict(
            range=[0, 10], gridcolor=COLOR_BORDER,
            title=dict(text="Score", font=dict(color=COLOR_TEXT_SECONDARY)),
            tickfont=dict(color=COLOR_TEXT_SECONDARY),
        ),
        xaxis=dict(
            tickangle=-30, gridcolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_TEXT_SECONDARY),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=COLOR_TEXT),
        ),
        height=380,
        **PLOTLY_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# AI Summary (frontend synthesis of existing report fields -- no new LLM call)
# ---------------------------------------------------------------------------

def render_ai_summary(report, audit_results, target_role_label: str) -> None:
    """
    Rendered as a single HTML string (CSS grid instead of st.columns) so the
    whole thing lives inside one real DOM node. The previous version opened
    <div class="ai-summary"> in one st.markdown() call, put the actual
    content in separate st.markdown()/st.columns() calls, and closed the div
    in a third call -- each st.markdown() becomes its own isolated element in
    Streamlit, so the div never wrapped anything. It rendered as an empty
    styled box with the real content appearing unstyled underneath it.

    v3: uses report.most_impressive_repo / best_repo_to_improve /
    recruiter_readiness_pct directly instead of independently recomputing
    "best repo" and a readiness estimate with different logic than
    core.pipeline's canonical calculation -- previously this box could
    show a different best/worst repo or a different readiness % than the
    Visual Report section below it, for the same underlying data.
    """
    skills_html = ""
    if report.gap_analysis.missing_skills:
        skills_preview = ", ".join(report.gap_analysis.missing_skills[:4])
        skills_html = f'<div class="ai-summary-full"><b>Skills to prioritize:</b> {skills_preview}</div>'

    st.markdown(
        f"""
        <div class="ai-summary">
            <h3>AI Summary</h3>
            <div class="ai-summary-grid">
                <div>
                    <p><b>Top strength:</b> {report.most_impressive_repo}</p>
                    <p><b>Recommended role:</b> {target_role_label}</p>
                </div>
                <div>
                    <p><b>Biggest gap:</b> {report.best_repo_to_improve}</p>
                    <p><b>Recruiter readiness:</b> {report.recruiter_readiness_pct}%</p>
                </div>
            </div>
            {skills_html}
        </div>
        """,
        unsafe_allow_html=True,
    )




# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

inject_css()


@st.cache_resource
def _get_cached_llm():
    # get_llm() reads HF_API_TOKEN via os.getenv -- Streamlit automatically
    # exposes root-level st.secrets keys as environment variables, so this
    # picks up .streamlit/secrets.toml locally and Streamlit Cloud's secrets
    # manager in production without any extra wiring here.
    return get_llm()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# Single source of truth for everything this app needs across Streamlit
# reruns. Every widget below either reads a field off `state` or writes
# one -- nothing lives in a separate, scattered session_state key except
# `target_role_select`, which Streamlit itself needs to own internally
# to manage the selectbox widget (see the comment further down).

DEFAULT_PIPELINE_STATE = {
    "phase": "input",             # "input" -> "audited" -> "done"
    "username": None,
    "target_role": None,
    "repos_data": None,
    "audit_results": None,
    "clarification_answers": {},
    "profile": None,
    "final_report": None,
    "confirmed_generate_anyway": False,
}

if "pipeline_state" not in st.session_state:
    st.session_state["pipeline_state"] = dict(DEFAULT_PIPELINE_STATE)

state = st.session_state["pipeline_state"]

inject_css()
llm = _get_cached_llm()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### AI GitHub Mentor")

    if state["username"]:
        st.markdown(f"**Current user:** `{state['username']}`")
    if state["audit_results"]:
        st.markdown(f"**Repositories:** {len(state['repos_data'])}")
    if state["final_report"]:
        st.markdown(f"**Portfolio score:** {state['final_report'].overall_score}/10")

    if state["phase"] != "input":
        if st.button("Start over", width="stretch"):
            st.session_state["pipeline_state"] = dict(DEFAULT_PIPELINE_STATE)
            st.session_state.pop("target_role_select", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Phase 1: input -- username, "Analyze Portfolio"
# ---------------------------------------------------------------------------

if state["phase"] == "input":
    st.markdown("## Analyze a GitHub Profile")
    st.caption("Enter a GitHub username to audit their public repositories.")

    raw_username = st.text_input("GitHub username", placeholder="e.g. octocat")
    analyze_clicked = st.button("Analyze Portfolio", width="stretch", type="primary")

    if analyze_clicked:
        if not raw_username.strip():
            render_popup("Enter a GitHub username first.", kind="warning")
        elif looks_like_url(raw_username) and not is_valid_github_profile_url(raw_username):
            render_popup(
                "That doesn't look like a GitHub profile URL. Try just the "
                "username, or a link like github.com/username.",
                kind="danger",
            )
        else:
            resolved_username = normalize_github_username(raw_username)
            if not check_username_exists(resolved_username):
                render_popup(f"GitHub user '{resolved_username}' doesn't seem to exist.", kind="danger")
            else:
                with st.spinner("Fetching repos and running the audit..."):
                    try:
                        repos_data, audit_results = run_audit_phase(
                            llm, resolved_username, max_repos=MAX_REPOS_DEFAULT,
                        )
                    except GitHubAPIError as error:
                        render_popup(str(error), kind="danger")
                        repos_data, audit_results = None, None
                    except Exception as error:
                        # Catches LLM-side failures (e.g. Groq rate limits) that
                        # GitHubAPIError doesn't cover -- without this, those
                        # errors crash the whole script with a raw traceback
                        # instead of a message the user can actually act on.
                        render_popup(
                            f"Something went wrong running the audit: {error}",
                            kind="danger",
                        )
                        repos_data, audit_results = None, None

                if repos_data is not None:
                    try:
                        profile = get_user_profile(resolved_username)
                    except GitHubAPIError:
                        profile = None  # display metadata only -- non-critical if this fails

                    state["username"] = resolved_username
                    state["repos_data"] = repos_data
                    state["audit_results"] = audit_results
                    state["profile"] = profile
                    state["clarification_answers"] = {}
                    state["phase"] = "audited"
                    st.rerun()


# ---------------------------------------------------------------------------
# Phase 2 & shared: audit results view (visible in both "audited" and "done")
# ---------------------------------------------------------------------------

if state["phase"] in ("audited", "done"):
    skills = extract_skills_from_repos(state["repos_data"])
    ai_signal_present = has_ai_domain_signal(state["repos_data"], skills)
    if not ai_signal_present:
        render_popup(
            "No AI/ML/data signal detected in these repos. This tool's skill-gap "
            "analysis, project suggestions, and roadmap are built entirely from "
            "AI/Data Science/ML job postings -- for a profile without that signal, "
            "those sections are likely to recommend irrelevant skills (e.g. "
            "TensorFlow) rather than anything useful for your actual focus area. "
            "The repository audit above is still valid regardless.",
            kind="warning",
        )

    st.markdown(f"### {state['username']}")

    # Target role selector, needed only once (before Generate Report),
    # but shown throughout for context. The pre-seed-before-first-render
    # trick below is the actual Streamlit lesson here: a widget's default
    # can only be set via its `index`/`value` argument (or by pre-seeding
    # its session_state key) BEFORE that widget has ever been instantiated.
    # Once instantiated, Streamlit forbids writing to that key directly --
    # v1 worked around this with a two-rerun "pending suggestion" dance.
    # Seeding it once here, guarded by "not already set", sidesteps that
    # entirely: after the first render, the widget's own key persists
    # whatever the user picks, and we never touch it again ourselves.
    if "target_role_select" not in st.session_state:
        suggested = suggest_target_role(state["repos_data"])
        suggested_label = next(
            (label for label, val in TARGET_ROLES.items() if val == suggested),
            "Any / no preference",
        )
        st.session_state["target_role_select"] = suggested_label

    role_label = st.selectbox("Target role", options=list(TARGET_ROLES.keys()), key="target_role_select")
    state["target_role"] = TARGET_ROLES[role_label]

    st.plotly_chart(skill_radar_chart(state["audit_results"]), config=PLOTLY_CONFIG, width="stretch")
    st.plotly_chart(repo_score_bar_chart(state["audit_results"]), config=PLOTLY_CONFIG, width="stretch")

    st.markdown("#### Repository Audit")
    # Lookup from repo name -> raw repo dict, so the audit-card loop below
    # can check possible_monorepo without changing AuditResult's schema --
    # that flag is computed in the tool layer (fetch_all_repo_data), not
    # something the LLM decides, so it doesn't belong on the LLM-produced
    # AuditResult object.
    repos_data_by_name = {r["name"]: r for r in state["repos_data"]}

    for repo in state["audit_results"]:
        with st.container(border=True):
            st.markdown(
                f"**{repo.repo_name}**  {score_badge(repo.doc_quality_score)}  {confidence_badge(repo.confidence)}",
                unsafe_allow_html=True,
            )
            if repos_data_by_name.get(repo.repo_name, {}).get("possible_monorepo"):
                render_popup(
                    f"'{repo.repo_name}' looks like it might contain multiple separate "
                    "projects as subfolders. This tool audits each GitHub repo as a "
                    "whole, not individual subfolders inside it -- the result below "
                    "reflects the whole repo, not each project inside it separately. "
                    "Consider splitting distinct projects into their own repos for a "
                    "more accurate audit (and generally, a stronger portfolio).",
                    kind="info",
                )
            st.caption(repo.notes)
            if repo.confidence == "low" and repo.clarifying_question:
                answer = st.text_input(
                    repo.clarifying_question,
                    key=f"clarify_{repo.repo_name}",
                    value=state["clarification_answers"].get(repo.repo_name, ""),
                )
                state["clarification_answers"][repo.repo_name] = answer

    if state["phase"] == "audited":
        generate_allowed = True
        if not ai_signal_present:
            state["confirmed_generate_anyway"] = st.checkbox(
                "I understand the sections below may not be relevant to my actual "
                "focus area -- generate the report anyway",
                value=state["confirmed_generate_anyway"],
            )
            generate_allowed = state["confirmed_generate_anyway"]

        if st.button("Generate Full Report", width="stretch", type="primary", disabled=not generate_allowed):
            with st.spinner("Running gap analysis and searching for contribution opportunities..."):
                try:
                    report = run_synthesis_phase(
                        llm, state["username"], state["repos_data"], state["audit_results"],
                        state["clarification_answers"], target_role=state["target_role"],
                        max_issues=MAX_ISSUES_DEFAULT,
                    )
                except Exception as error:
                    render_popup(f"Something went wrong generating the report: {error}", kind="danger")
                    report = None

            if report is not None:
                state["final_report"] = report
                state["phase"] = "done"
                st.rerun()


# ---------------------------------------------------------------------------
# Phase 3: done -- final report, download, contribution opportunities
# ---------------------------------------------------------------------------

if state["phase"] == "done":
    report = state["final_report"]
    target_role_label = next(
        (label for label, val in TARGET_ROLES.items() if val == state["target_role"]),
        "Any / no preference",
    )

    st.markdown("## Final Report")
    render_ai_summary(report, state["audit_results"], target_role_label=target_role_label)
    st.markdown(report.summary)

    # --- Visual report -- text-based bars, computed from real data ---
    # Missing Skills bar deliberately omitted (no agreed-on percentage
    # definition for it yet -- explicit decision, not an oversight).
    st.markdown("### Visual Report")
    bar_cols = st.columns(3)
    with bar_cols[0]:
        st.text(render_bar("Portfolio Score", report.overall_score * 10))
    with bar_cols[1]:
        st.text(render_bar("Documentation", report.documentation_avg * 10))
    with bar_cols[2]:
        st.text(render_bar("Recruiter Readiness", report.recruiter_readiness_pct))

    st.markdown("### Strengths")
    for item in report.strengths:
        st.markdown(f"- {item}")

    st.markdown("### Weaknesses")
    for item in report.weaknesses:
        st.markdown(f"- {item}")

    st.markdown("### Most Impressive Repository")
    st.markdown(report.most_impressive_repo)

    st.markdown("### Best Repository to Improve")
    st.markdown(report.best_repo_to_improve)

    st.markdown("### Top Missing Technologies")
    for skill in report.top_missing_technologies:
        st.markdown(f"- {skill}")

    st.markdown("### Recommended Learning Order")
    for i, skill in enumerate(report.recommended_learning_order, 1):
        st.markdown(f"{i}. {skill}")

    st.markdown("### Recommended Portfolio Projects")
    for project in report.gap_analysis.suggested_projects:
        with st.container(border=True):
            st.markdown(f"**{project.title}**")
            st.caption(f"Uses: {', '.join(project.tech_stack)}")
            st.markdown(f"*Real-world challenge:* {project.real_world_challenge}")

    # is_match is used here directly -- the structured schema field we
    # added when we rebuilt contribution_filter_chain means this can
    # filter reliably, instead of v1's approach of guessing yes/no from
    # a free-text relevance_reason sentence.
    real_matches = [m for m in report.contributions if m.is_match]
    st.markdown("### Recommended Open-Source Issues")
    if real_matches:
        for match in real_matches:
            # repo_url is the GitHub API form (api.github.com/repos/owner/repo) --
            # extracted here purely for display, so identically-titled issues
            # from different repos (common with bot-filed issues like "Fix
            # pytest deprecation warnings") are distinguishable in the report.
            repo_name = match.repo_url.replace("https://api.github.com/repos/", "")
            st.markdown(f"- [{match.issue_title}]({match.issue_url}) -- *{repo_name}* -- {match.relevance_reason}")
    else:
        st.caption("No strong contribution matches found for this search.")

    st.markdown("### 90-Day Roadmap")
    for i, item in enumerate(report.roadmap_90_day, 1):
        st.markdown(f"{i}. {item}")

    st.download_button(
        "Download Report (Markdown)",
        data=report.to_markdown(),
        file_name=f"{state['username']}_report.md",
        mime="text/markdown",
        width="stretch",
    )
