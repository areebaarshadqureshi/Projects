from langchain_core.prompts import ChatPromptTemplate

REPO_AUDIT_TEMPLATE = """You are reviewing a GitHub repository for a portfolio audit.

Repository name: {name}
Description: {description}
Primary language: {language}
Topics: {topics}
Commit count: {commit_count}
README content:
{readme}

Score the README quality and folder structure from 1 to 10.
If you cannot clearly tell what this project does or how to run it from the README alone,
set confidence to "low" and write one specific clarifying question about this exact repo.
If the purpose is clear, set confidence to "high" and leave the question blank.

{format_instructions}
"""

repo_audit_prompt = ChatPromptTemplate.from_template(REPO_AUDIT_TEMPLATE)
