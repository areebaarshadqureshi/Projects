from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from utils.github_api_client import (
    get_user_repos,
    get_repo_readme,
    get_repo_commits,
    get_repo_root_contents,
    normalize_github_username,
)

# Common non-project top-level folder names -- excluded when counting
# "project-like" subdirectories, so a normal repo with src/tests/docs
# doesn't get wrongly flagged as a monorepo of separate projects.
_NON_PROJECT_DIR_NAMES = {
    "src", "test", "tests", "docs", "doc", ".github", ".vscode", ".idea",
    "assets", "static", "public", "scripts", "examples", "example",
    "dist", "build", "node_modules", "__pycache__", ".git", "bin", "lib",
    "vendor", "config", "configs",
}
_MONOREPO_SUBDIR_THRESHOLD = 3


def _looks_like_monorepo(root_contents: list[dict]) -> bool:
    """
    Soft heuristic, not a hard classifier -- flags a repo as a possible
    monorepo (several unrelated projects living as subfolders of one
    repo, e.g. a "Projects" repo) if it has several top-level
    directories that AREN'T common non-project folder names like src/
    or tests/. This can false-positive on repos with genuinely many
    real subpackages, and false-negative on a monorepo that happens to
    use only 1-2 subfolders -- it's meant to prompt a human to check,
    not to be authoritative.
    """
    project_like_dirs = [
        item for item in root_contents
        if item.get("type") == "dir" and item.get("name", "").lower() not in _NON_PROJECT_DIR_NAMES
    ]
    return len(project_like_dirs) >= _MONOREPO_SUBDIR_THRESHOLD


class RepoFetchInput(BaseModel):
    username: str = Field(description="GitHub username to analyze")
    max_repos: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum number of repos to fetch, most recently updated first. "
            "Caps GitHub API usage and keeps the audit focused on active work "
            "rather than every repo a user has ever created."
        ),
    )


def fetch_all_repo_data(username: str, max_repos: int = 20) -> list[dict]:
    username = normalize_github_username(username)
    repos = get_user_repos(username, max_repos=max_repos)
    enriched = []
    for repo in repos:
        readme = get_repo_readme(username, repo["name"])
        commits = get_repo_commits(username, repo["name"], limit=10)
        root_contents = get_repo_root_contents(username, repo["name"])
        enriched.append({
            "name": repo["name"],
            "description": repo.get("description") or "",
            "language": repo.get("language") or "",
            "topics": repo.get("topics", []),
            "readme": readme,
            "commit_count": len(commits),
            "url": repo["html_url"],
            "possible_monorepo": _looks_like_monorepo(root_contents),
        })
    return enriched


github_repo_tool = StructuredTool.from_function(
    func=fetch_all_repo_data,
    name="fetch_github_repos",
    description=(
        "Fetches up to max_repos public repos (most recently updated first), "
        "along with their READMEs and commit history, for a GitHub username."
    ),
    args_schema=RepoFetchInput,
)
