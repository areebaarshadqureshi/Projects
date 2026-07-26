from pydantic import BaseModel, Field


class PortfolioNarrative(BaseModel):
    strengths: list[str] = Field(
        description="2-4 concrete strengths, each grounded in specific repo evidence "
        "(e.g. 'Clear README with setup instructions in X'), not generic praise"
    )
    weaknesses: list[str] = Field(
        description="2-4 concrete, actionable weaknesses (e.g. 'No tests in Y'), "
        "not generic criticism"
    )
    recommended_learning_order: list[str] = Field(
        description="The missing skills reordered into a sensible learning sequence "
        "(foundational skills first), 3-6 items, short skill names only"
    )
    roadmap_90_day: list[str] = Field(
        description="8-10 flat, concrete action items covering the next 90 days, "
        "sequenced but not tied to specific days/weeks (e.g. 'Add automated tests "
        "to your top repo', not 'Day 1-7: write tests')"
    )
    recruiter_readiness_pct: int = Field(
        ge=0, le=100,
        description="Overall estimate (0-100) of how ready this portfolio is for a "
        "recruiter to review right now, weighing documentation quality, code "
        "structure, and how large the skill gaps are, together"
    )
