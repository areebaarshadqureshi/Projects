from pydantic import BaseModel
from schemas.audit_result import AuditResult
from schemas.gap_analysis_result import GapAnalysisResult
from schemas.contribution_match import ContributionMatch


class FinalReport(BaseModel):
    username: str
    overall_score: float
    documentation_avg: float
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    most_impressive_repo: str
    best_repo_to_improve: str
    top_missing_technologies: list[str]
    recommended_learning_order: list[str]
    roadmap_90_day: list[str]
    recruiter_readiness_pct: int
    audit_results: list[AuditResult]
    gap_analysis: GapAnalysisResult
    contributions: list[ContributionMatch]

    def to_markdown(self) -> str:
        lines = [
            f"# GitHub Portfolio Report — {self.username}",
            f"\n## Overall Portfolio Score: {self.overall_score} / 10\n",
            f"{self.summary}\n",
            "## Strengths\n",
        ]
        for item in self.strengths:
            lines.append(f"- {item}")
        lines.append("\n## Weaknesses\n")
        for item in self.weaknesses:
            lines.append(f"- {item}")
        lines.append(f"\n## Most Impressive Repository\n\n{self.most_impressive_repo}\n")
        lines.append(f"## Best Repository to Improve\n\n{self.best_repo_to_improve}\n")
        lines.append("## Repo-by-Repo Findings\n")
        for repo in self.audit_results:
            lines.append(f"### {repo.repo_name}")
            lines.append(f"- Documentation Quality: {repo.doc_quality_score}/10")
            lines.append(f"- Structure: {repo.structure_score}/10")
            lines.append(f"- Confidence: {repo.confidence}")
            lines.append(f"- Notes: {repo.notes}\n")
        lines.append("## Top Missing Technologies\n")
        for skill in self.top_missing_technologies:
            lines.append(f"- {skill}")
        lines.append("\n## Recommended Learning Order\n")
        for i, skill in enumerate(self.recommended_learning_order, 1):
            lines.append(f"{i}. {skill}")
        lines.append("\n## Recommended Portfolio Projects\n")
        for project in self.gap_analysis.suggested_projects:
            lines.append(f"### {project.title}")
            lines.append(f"**Uses:** {', '.join(project.tech_stack)}")
            lines.append(f"**Real-world challenge:** {project.real_world_challenge}\n")
        lines.append("## Recommended Open-Source Issues\n")
        for match in self.contributions:
            if match.is_match:
                repo_name = match.repo_url.replace("https://api.github.com/repos/", "")
                lines.append(f"- [{match.issue_title}]({match.issue_url}) — *{repo_name}* — {match.relevance_reason}")
        lines.append("\n## 90-Day Roadmap\n")
        for i, item in enumerate(self.roadmap_90_day, 1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)
