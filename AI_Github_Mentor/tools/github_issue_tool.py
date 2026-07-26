from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from utils.github_api_client import search_good_first_issues


class IssueSearchInput(BaseModel):
    language: str = Field(description="Primary programming language to filter by")
    max_issues: int = Field(
        default=10,
        ge=1,
        le=30,
        description=(
            "Maximum number of good-first-issues to return. Capped at 30 -- "
            "beyond that, a report becomes noise rather than a useful shortlist."
        ),
    )


github_issue_tool = StructuredTool.from_function(
    func=search_good_first_issues,
    name="search_good_first_issues",
    description="Searches GitHub for open good-first-issue labeled issues in a given language",
    args_schema=IssueSearchInput,
)
