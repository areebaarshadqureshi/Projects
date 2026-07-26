from langchain_core.prompts import ChatPromptTemplate

FILTER_TEMPLATE = """The user's skills: {skills}

Candidate GitHub issue:
Repo: {repo_url}
Title: {issue_title}
URL: {issue_url}

Decide whether this issue is a realistic, relevant match for someone with
these skills. Set is_match to true or false accordingly, and give one
sentence explaining your reasoning in relevance_reason either way.

{format_instructions}
"""

filter_prompt = ChatPromptTemplate.from_template(FILTER_TEMPLATE)
