from langchain_core.output_parsers import PydanticOutputParser
from schemas.contribution_match import ContributionMatch
from prompts.contribution_filter_prompt import filter_prompt
from utils.llm_output_cleaning import clean_llm_json_output


def build_contribution_filter_chain(llm):
    parser = PydanticOutputParser(pydantic_object=ContributionMatch)
    return (
        filter_prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | clean_llm_json_output
        | parser
    )


def filter_issue(llm, skills: list[str], issue: dict) -> ContributionMatch:
    chain = build_contribution_filter_chain(llm)
    return chain.invoke({
        "skills": ", ".join(skills),
        "repo_url": issue["repository_url"],
        "issue_title": issue["title"],
        "issue_url": issue["html_url"],
    })


def run_contribution_search(llm, skills: list[str], language: str, max_issues: int = 10) -> list[ContributionMatch]:
    """
    Fetches up to max_issues good-first-issues for `language`, then runs
    every fetched issue through the filter chain (no more hardcoded [:5] --
    the caller controls cost entirely via max_issues, which now actually
    reaches the tool call it's meant to control).
    """
    from tools.github_issue_tool import github_issue_tool
    issues = github_issue_tool.invoke({"language": language, "max_issues": max_issues})

    if not issues:
        return []

    # Dedup by html_url (the true unique identifier) -- protects against
    # the same issue literally appearing twice in the search results.
    seen_urls = set()
    deduped_issues = []
    for issue in issues:
        if issue["html_url"] not in seen_urls:
            seen_urls.add(issue["html_url"])
            deduped_issues.append(issue)
    issues = deduped_issues

    # Also dedup by title (keep first occurrence only). Originally left
    # this out deliberately -- different repos can legitimately share a
    # bot-filed title. But in practice, most same-titled duplicates turn
    # out to be forked/cloned copies of the same course or template repo
    # (e.g. four different accounts all forking the same "recipe-explorer"
    # bootcamp project, each carrying the same auto-generated issue).
    # Showing that same trivial issue 4x adds nothing -- a real second
    # opportunity is worth more than proving the search technically found
    # more results.
    seen_titles = set()
    title_deduped_issues = []
    for issue in issues:
        if issue["title"] not in seen_titles:
            seen_titles.add(issue["title"])
            title_deduped_issues.append(issue)
    issues = title_deduped_issues

    chain = build_contribution_filter_chain(llm)
    inputs = [{
        "skills": ", ".join(skills),
        "repo_url": issue["repository_url"],
        "issue_title": issue["title"],
        "issue_url": issue["html_url"],
    } for issue in issues]

    
    return chain.batch(inputs, config={"max_concurrency": 2})
