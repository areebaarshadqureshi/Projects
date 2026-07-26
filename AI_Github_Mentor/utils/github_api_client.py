import os
import re
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "Accept": "application/vnd.github+json",
}
BASE_URL = "https://api.github.com"

# Every call in this module used to omit `timeout=`, which for the requests
# library means "wait forever." On a flaky connection that turns into an
# indefinite hang during run_audit_phase (per-repo commits/readme fetches
# in a loop) instead of a fast, recoverable failure. 10s is generous for a
# single GitHub REST call; connect vs. read are split so a slow-to-connect
# network fails faster than a slow-to-respond (but connected) one.
REQUEST_TIMEOUT = (5, 10)  # (connect timeout, read timeout), seconds


class GitHubAPIError(Exception):
    """
    Raised for any GitHub API call that fails after normalization/status
    handling below -- network errors (timeout, DNS, connection refused) and
    non-2xx responses alike. Callers (orchestrator/streamlit) catch this one
    type instead of needing to know about requests.exceptions internals.
    """
    def __init__(self, message: str, username: str | None = None, repo: str | None = None):
        self.username = username
        self.repo = repo
        super().__init__(message)


def normalize_github_username(raw: str) -> str:
    """
    Accepts either a bare username ("areebaarshadqureshi") or a pasted
    GitHub profile URL (e.g. "https://github.com/areebaarshadqureshi",
    "github.com/areebaarshadqureshi/", or
    "https://github.com/areebaarshadqureshi?tab=repositories") and
    returns just the username.

    Without this, a pasted URL gets passed straight into the API path
    (e.g. /users/https://github.com/areebaarshadqureshi/repos), which
    GitHub's API 404s on.
    """
    if not raw:
        return raw

    username = raw.strip()

    # Strip protocol + domain if present ("https://github.com/", "github.com/")
    username = re.sub(r"^(https?://)?(www\.)?github\.com/", "", username, flags=re.IGNORECASE)

    # Drop any query string or path segments after the username
    # (?tab=repositories, /repos, trailing slash, etc.)
    username = username.split("?")[0].split("/")[0]

    return username.strip()


def looks_like_url(raw: str) -> bool:
    return bool(re.match(r"^(https?://|www\.)", raw.strip(), flags=re.IGNORECASE))


def is_valid_github_profile_url(raw: str) -> bool:
    """
    A GitHub *profile* URL specifically -- github.com/<user>, no repo path,
    no github.io, no gist.github.com, etc.
    """
    pattern = r"^(https?://)?(www\.)?github\.com/[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/?(\?.*)?$"
    return bool(re.match(pattern, raw.strip(), flags=re.IGNORECASE))


def check_username_exists(username: str) -> bool:
    """
    Lightweight existence check, separate from get_user_profile() so a UI
    can give a fast, specific error before kicking off the full audit.
    Moved here from app/streamlit_app.py in v2 -- this makes a live
    GitHub API call, which has no business living in the UI file per the
    backend/frontend boundary decided in Stage 1.
    """
    try:
        resp = requests.get(f"{BASE_URL}/users/{username}", headers=HEADERS, timeout=(5, 10))
    except requests.RequestException:
        # Network hiccup on *this* quick check -- don't hard-block here;
        # the real audit call right after will surface a clear error
        # (via GitHubAPIError) if GitHub is genuinely unreachable.
        return True

    if resp.status_code == 404:
        return False
    if resp.status_code == 200:
        return True
    # Anything else (403 rate-limited, 5xx, etc.) is NOT the same as "this
    # user doesn't exist" -- fail open here and let the real audit call
    # right after surface a specific, accurate error instead of this quick
    # pre-check silently misreporting a rate limit as a missing username.
    return True


def get_user_profile(username: str) -> dict:
    """
    Fetches the public GitHub profile (avatar, name, bio, follower counts,
    public repo count) for the profile header card in the UI. This is
    display metadata only -- it plays no role in the audit/gap-analysis
    pipeline, which continues to work purely off get_user_repos() and the
    per-repo README/commit data.
    """
    username = normalize_github_username(username)
    url = f"{BASE_URL}/users/{username}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise GitHubAPIError(
            f"GitHub API timed out fetching the profile for '{username}'. "
            "GitHub may be slow to respond, or your connection dropped -- try again.",
            username=username,
        )
    except requests.exceptions.HTTPError:
        if response.status_code == 404:
            raise GitHubAPIError(f"GitHub user '{username}' doesn't exist.", username=username)
        raise GitHubAPIError(
            f"GitHub API returned an error ({response.status_code}) for '{username}'.",
            username=username,
        )
    except requests.exceptions.RequestException as exc:
        raise GitHubAPIError(f"Couldn't reach GitHub while fetching '{username}': {exc}", username=username)
    data = response.json()
    return {
        "login": data.get("login", username),
        "name": data.get("name") or data.get("login", username),
        "avatar_url": data.get("avatar_url", ""),
        "bio": data.get("bio") or "",
        "followers": data.get("followers", 0),
        "following": data.get("following", 0),
        "public_repos": data.get("public_repos", 0),
        "html_url": data.get("html_url", f"https://github.com/{username}"),
    }


def get_user_repos(username: str, max_repos: int = 20) -> list[dict]:
    """
    Fetches up to `max_repos` public repos for a username, sorted by most
    recently updated first. Sorting server-side (not fetching everything
    and slicing in Python) matters for two reasons: it saves GitHub API
    rate-limit budget on accounts with many repos, and it means the repos
    we actually audit are the ones most likely to reflect current skill --
    not whatever order the API's default (alphabetical by full_name)
    happens to return.
    """
    username = normalize_github_username(username)
    url = f"{BASE_URL}/users/{username}/repos"
    params = {
        "per_page": max_repos,
        "sort": "updated",
        "direction": "desc",
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise GitHubAPIError(
            f"GitHub API timed out fetching repositories for '{username}'. Try again in a moment.",
            username=username,
        )
    except requests.exceptions.HTTPError:
        if response.status_code == 404:
            raise GitHubAPIError(f"GitHub user '{username}' doesn't exist.", username=username)
        raise GitHubAPIError(
            f"GitHub API returned an error ({response.status_code}) listing repos for '{username}'.",
            username=username,
        )
    except requests.exceptions.RequestException as exc:
        raise GitHubAPIError(f"Couldn't reach GitHub while listing repos for '{username}': {exc}", username=username)
    return response.json()


def get_repo_readme(username: str, repo: str) -> str:
    url = f"{BASE_URL}/repos/{username}/{repo}/readme"
    try:
        response = requests.get(
            url, headers={**HEADERS, "Accept": "application/vnd.github.raw"}, timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException:
        # This runs inside a per-repo loop (fetch_all_repo_data) -- one repo's
        # transient network hiccup shouldn't abort the audit for every other
        # repo. Degrade to "no README" for this repo and let the loop continue;
        # get_user_repos() above already proved GitHub is reachable overall.
        return ""
    if response.status_code == 404:
        return ""
    response.raise_for_status()
    return response.text


def get_repo_commits(username: str, repo: str, limit: int = 30) -> list[dict]:
    url = f"{BASE_URL}/repos/{username}/{repo}/commits"
    try:
        response = requests.get(url, headers=HEADERS, params={"per_page": limit}, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        # Same reasoning as get_repo_readme: degrade this one repo rather
        # than aborting the whole audit loop.
        return []
    if response.status_code == 409:  # empty repo
        return []
    response.raise_for_status()
    return response.json()


def get_repo_root_contents(username: str, repo: str) -> list[dict]:
    """
    Lists the top-level files/folders of a repo -- used to heuristically
    flag possible monorepos (a repo containing several unrelated projects
    as subfolders, e.g. a "Projects" repo with each real project as its
    own subdirectory). Same degrade-gracefully pattern as
    get_repo_readme/get_repo_commits: this runs inside the per-repo loop
    in fetch_all_repo_data, so a failure here shouldn't abort auditing
    every other repo.
    """
    url = f"{BASE_URL}/repos/{username}/{repo}/contents"
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        return []
    if response.status_code != 200:
        return []
    return response.json()


def search_good_first_issues(language: str, max_issues: int = 10) -> list[dict]:
    url = f"{BASE_URL}/search/issues"
    query = f"is:issue label:good-first-issue state:open language:{language}"
    try:
        response = requests.get(
            url, headers=HEADERS, params={"q": query, "per_page": max_issues}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise GitHubAPIError(f"GitHub API timed out searching good-first-issues for '{language}'.")
    except requests.exceptions.RequestException as exc:
        raise GitHubAPIError(f"Couldn't reach GitHub while searching issues for '{language}': {exc}")
    return response.json().get("items", [])
