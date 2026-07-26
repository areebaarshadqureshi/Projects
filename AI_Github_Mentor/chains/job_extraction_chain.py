"""
chains/job_extraction_chain.py

Takes a raw search result (title + snippet + url) and extracts it into
the same structured format used in data/job_requirements/*.json.

Snippets from search results are short, so the extraction will sometimes
be incomplete (a snippet might not mention every required skill the full
posting has). Treat this as a fast first pass, not a perfect substitute
for reading the full posting when something looks important.
"""

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class ExtractedJobRequirement(BaseModel):
    role: str = Field(description="Normalized job title, e.g. 'Data Scientist'")
    seniority: Literal["entry-level", "entry-mid", "mid", "senior"] = Field(
        description="Best guess at seniority based on the snippet"
    )
    required_skills: list[str] = Field(
        description="Skills explicitly mentioned as required -- each entry must be a "
        "short technology or skill NAME (1-4 words, e.g. 'PowerBI', 'REST APIs', "
        "'Docker'), never a full sentence or responsibility description"
    )
    nice_to_have: list[str] = Field(
        description="Skills mentioned as a plus or preferred, not required -- same "
        "short-tag format as required_skills"
    )
    source_url: str
    notes: str = Field(description="One sentence on anything notable about this listing")
    is_karachi_relevant: bool = Field(
        description="True if the posting is for Karachi or clearly Pakistan-market relevant"
    )


EXTRACTION_TEMPLATE = """You are extracting structured data from a job posting search result.

Title: {title}
Snippet: {snippet}
URL: {url}

Extract the role, seniority, required skills, and nice-to-have skills.
Each skill entry must be a short technology or skill NAME only (1-4 words,
e.g. "PowerBI", "REST APIs", "Docker") -- never a full sentence describing
a responsibility or duty. If the snippet describes a duty rather than
naming a specific skill (e.g. "creating dashboards and reports"), extract
the underlying tool/technology name if one is implied, or omit it entirely
rather than including the sentence as-is.
If the snippet is too short or vague to determine something confidently,
make a reasonable best guess and say so in the notes field, rather than
inventing specific skills that are not implied by the text.
Set is_karachi_relevant to false if this is clearly not a Pakistan-market listing.

{format_instructions}
"""

extraction_prompt = ChatPromptTemplate.from_template(EXTRACTION_TEMPLATE)

from utils.llm_output_cleaning import clean_llm_json_output

def build_extraction_chain(llm):
    parser = PydanticOutputParser(pydantic_object=ExtractedJobRequirement)
    return (
        extraction_prompt.partial(
            format_instructions=parser.get_format_instructions()
        )
        | llm
        | clean_llm_json_output
        | parser
    )


def extract_job_requirement(llm, search_result: dict) -> ExtractedJobRequirement:
    chain = build_extraction_chain(llm)
    return chain.invoke({
        "title": search_result["title"],
        "snippet": search_result["snippet"],
        "url": search_result["url"],
    })
