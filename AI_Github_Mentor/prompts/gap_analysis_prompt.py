"""
Prompt for the Gap Analysis Chain (RAG step, Phase 3).

v3: suggested_projects tightened to avoid generic ideas (v2's model
kept suggesting things like "Build a CRUD API" -- technically correct,
useless as a portfolio differentiator). Now explicitly asks for a
named tech stack and a concrete "real-world challenge" the project
demonstrates solving, with a worked example of the difference.
"""

from langchain_core.prompts import ChatPromptTemplate

GAP_ANALYSIS_TEMPLATE = """The user's demonstrated skills, from their GitHub repos: {skills}

Target role: {target_role}

Relevant job requirements found for this target role:
{retrieved_requirements}

List skills that appear in the job requirements but are missing from the user's skills.
Each entry in missing_skills must be a short technology or skill NAME only
(1-4 words, e.g. "Docker", "REST APIs") -- if the job requirements text
contains a full sentence describing a duty rather than naming a specific
skill, extract just the underlying technology/skill name from it, never
copy the sentence as-is.

Then suggest 2 to 3 concrete portfolio projects that would close those gaps.
Each project needs a specific title, a tech_stack list (3-6 real
technologies), and a real_world_challenge sentence naming the actual
engineering problem it demonstrates solving.

AVOID generic, low-differentiation ideas. Compare:
  Weak:   "Build a CRUD API"
  Strong: title: "Distributed Event Ticketing System"
          tech_stack: ["Docker", "Redis", "RabbitMQ", "PostgreSQL", "CI/CD"]
          real_world_challenge: "Handle concurrent ticket purchases without overselling"
The strong version is specific, names real infrastructure, and states an
actual hard problem -- not just "build X using Y". Match that level of
specificity for every suggested project.

{format_instructions}
"""

gap_analysis_prompt = ChatPromptTemplate.from_template(GAP_ANALYSIS_TEMPLATE)
