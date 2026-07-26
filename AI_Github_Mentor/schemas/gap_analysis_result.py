from pydantic import BaseModel, Field


class SuggestedProject(BaseModel):
    title: str = Field(description="A specific, impressive project name -- not generic ('Build a CRUD API')")
    tech_stack: list[str] = Field(
        description="Concrete technologies/tools this project would use, e.g. "
        "['Docker', 'Redis', 'RabbitMQ', 'PostgreSQL', 'CI/CD']. 3-6 items."
    )
    real_world_challenge: str = Field(
        description="One sentence naming the concrete engineering problem this project "
        "demonstrates solving (e.g. 'Handle concurrent ticket purchases without "
        "overselling'), not a restatement of the title"
    )


class GapAnalysisResult(BaseModel):
    missing_skills: list[str] = Field(
        description="Skills required by target roles but absent from the user's repos -- "
        "each entry must be a short technology or skill NAME (1-4 words, e.g. 'Docker', "
        "'REST APIs'), never a full sentence or job-responsibility description"
    )
    suggested_projects: list[SuggestedProject] = Field(description="Concrete project ideas to close each gap")
