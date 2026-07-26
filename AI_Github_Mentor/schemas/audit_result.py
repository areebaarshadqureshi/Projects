from pydantic import BaseModel, Field
from typing import Literal


class AuditResult(BaseModel):
    repo_name: str = Field(description="Name of the repository")
    doc_quality_score: int = Field(description="README quality score, 1 to 10")
    structure_score: int = Field(description="Folder/file structure score, 1 to 10")
    confidence: Literal["high", "low"] = Field(
        description="high if the reviewer understands the repo's purpose, low if unclear"
    )
    notes: str = Field(description="Short explanation of the score")
    clarifying_question: str = Field(
        default="", description="Only filled in if confidence is low"
    )
