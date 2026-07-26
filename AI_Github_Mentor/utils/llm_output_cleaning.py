"""
utils/llm_output_cleaning.py

Defensive cleaning for structured LLM output before it reaches
PydanticOutputParser. Smaller open-source instruct models don't always
stop cleanly after the JSON they were asked for -- in practice we've
seen three failure modes:

  1. Wrapping the JSON in a markdown code fence (```json ... ```)
  2. Rambling past the JSON with extra hallucinated text -- a fake
     "Assistant:" turn, a second "corrected" JSON block, prose
     commentary second-guessing the answer. We take only the FIRST
     complete JSON object and discard everything after it.
  3. Using Python's capitalized True/False/None instead of JSON's
     lowercase true/false/null.

Any chain piping `llm | parser` directly (PydanticOutputParser) is
vulnerable to all three. Insert clean_llm_json_output as a step between
the LLM and the parser: `... | llm | clean_llm_json_output | parser`.
"""

import re

_PY_LITERAL_PATTERN = re.compile(r'([:\[,]\s*)(True|False|None)(?=\s*[,\]\}])')
_PY_LITERAL_MAP = {"True": "true", "False": "false", "None": "null"}


def _normalize_python_literals(text: str) -> str:
    """
    Only replaces True/False/None when they appear in an actual JSON
    value position (right after `:`, `[`, or `,`) -- deliberately NOT a
    blanket word-replace, since that would also corrupt a legitimate
    string value that happens to contain the English word "True", e.g.
    a relevance_reason like "...this is True for advanced roles only".
    """
    def _replace(match: re.Match) -> str:
        return match.group(1) + _PY_LITERAL_MAP[match.group(2)]
    return _PY_LITERAL_PATTERN.sub(_replace, text)


def _extract_first_json_block(text: str) -> str:
    """
    Returns only the FIRST complete top-level JSON object in the text,
    discarding anything before or after it -- this is what protects
    against a model that rambles a second "corrected" block or trailing
    prose after the JSON it was actually asked for.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # No fence -- find the first balanced {...} by brace-counting rather
    # than a greedy regex, since a naive `\{.*\}` would swallow a second
    # trailing JSON object into one invalid blob.
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # unbalanced -- return what we have, let the parser raise a clear error


def clean_llm_json_output(llm_output) -> str:
    """
    Accepts either a plain string (from a completion-style LLM) or an
    AIMessage (from a ChatModel like ChatGroq/ChatHuggingFace -- both
    return AIMessage from .invoke(), not a string). PydanticOutputParser
    unwraps AIMessage.content automatically; a plain function inserted
    into the chain via RunnableLambda does not, so this has to do it
    manually or every chat-model-backed chain crashes here.
    """
    text = llm_output.content if hasattr(llm_output, "content") else llm_output
    return _normalize_python_literals(_extract_first_json_block(text))
