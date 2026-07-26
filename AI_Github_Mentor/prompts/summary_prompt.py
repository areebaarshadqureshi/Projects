from langchain_core.prompts import ChatPromptTemplate

SUMMARY_TEMPLATE = """Write a short, encouraging 3 sentence summary of this GitHub portfolio review.
Overall score: {overall_score}/10
Number of repos reviewed: {repo_count}
Top missing skill: {top_missing_skill}
Keep it plain and direct, no exaggerated praise, no em dashes.
"""

summary_prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
