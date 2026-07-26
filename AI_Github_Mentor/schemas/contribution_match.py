from pydantic import BaseModel, Field


class ContributionMatch(BaseModel):
    repo_url: str
    issue_title: str
    issue_url: str
    is_match: bool = Field(
        description="Whether this issue is a realistic, relevant match for the user's skill set"
    )
    relevance_reason: str = Field(
        description="One sentence explaining why this is or isn't a good match"
    )
