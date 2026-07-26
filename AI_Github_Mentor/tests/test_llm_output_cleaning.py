"""
Tests for utils/llm_output_cleaning.py.

test_clean_llm_json_output_reproduces_contribution_filter_bug is a
near-verbatim reproduction of the actual production traceback: a model
writing "is_match": False (capitalized), then rambling a fake
"Assistant:" turn with a second, self-corrected JSON block afterward.
"""

from utils.llm_output_cleaning import clean_llm_json_output


def test_strips_markdown_json_fence():
    raw = '```json\n{"role": "Data Scientist"}\n```'
    assert clean_llm_json_output(raw) == '{"role": "Data Scientist"}'


def test_finds_raw_json_without_fence():
    raw = 'Sure, here you go: {"role": "Data Scientist"} -- hope that helps!'
    assert clean_llm_json_output(raw) == '{"role": "Data Scientist"}'


def test_passes_through_unparseable_text_unchanged():
    """
    No JSON found at all -- pass through as-is and let PydanticOutputParser
    raise its own clear error, rather than this function masking the
    problem with a confusing transformation.
    """
    raw = "no json here at all"
    assert clean_llm_json_output(raw) == raw


def test_handles_aimessage_input_not_just_plain_strings():
    """
    Regression test: ChatGroq/ChatHuggingFace return AIMessage from
    .invoke(), not a plain string. A RunnableLambda-wrapped plain
    function doesn't auto-unwrap that the way PydanticOutputParser does
    -- this crashed with TypeError: expected string or bytes-like
    object, got 'AIMessage' before content-unwrapping was added.
    """
    from langchain_core.messages import AIMessage

    message = AIMessage(content='```json\n{"role": "Data Scientist"}\n```')
    assert clean_llm_json_output(message) == '{"role": "Data Scientist"}'


def test_normalizes_python_true_false_none():
    raw = '{"is_match": False, "confidence": None, "flagged": True}'
    cleaned = clean_llm_json_output(raw)
    assert cleaned == '{"is_match": false, "confidence": null, "flagged": true}'


def test_does_not_corrupt_string_values_containing_true_or_false():
    """
    The word "True" inside an actual string value (not a JSON boolean
    position) must NOT be touched -- this is the whole reason the
    normalization regex requires True/False/None to sit in an actual
    value position (after :, [, or ,), not just anywhere in the text.
    """
    raw = '{"relevance_reason": "This is True for advanced roles only"}'
    assert clean_llm_json_output(raw) == raw


def test_extracts_only_first_json_block_when_model_rambles_a_second_one():
    """
    A naive greedy regex (\\{.*\\}) would swallow BOTH JSON blocks below
    into one invalid blob. This proves only the first, complete object
    is extracted.
    """
    raw = '''Here is the first attempt:
{"a": 1, "b": 2}
Wait, let me reconsider:
{"a": 3, "b": 4}
'''
    assert clean_llm_json_output(raw) == '{"a": 1, "b": 2}'


def test_clean_llm_json_output_reproduces_contribution_filter_bug():
    """
    Near-verbatim reproduction of the actual production traceback:
    a fenced block with "is_match": False (capitalized), followed by a
    fake "Assistant:" turn, a second self-corrected block, and trailing
    prose. Proves the fix actually resolves the real failure, not just
    a simplified version of it.
    """
    import json

    raw = '''```
{
  "repo_url": "https://api.github.com/repos/mohitkumhar/business-ai-agent",
  "issue_title": "[DOCS] Audit 310: Add a docstring for public function import_notebook",
  "issue_url": "https://github.com/mohitkumhar/business-ai-agent/issues/375",
  "is_match": False,
  "relevance_reason": "This issue is about adding documentation."
}
```

Assistant: ```json
{
  "repo_url": "https://api.github.com/repos/mohitkumhar/business-ai-agent",
  "issue_title": "[DOCS] Audit 310: Add a docstring for public function import_notebook",
  "issue_url": "https://github.com/mohitkumhar/business-ai-agent/issues/375",
  "is_match": True,
  "relevance_reason": "Adding a docstring is a documentation task."
}
```

Note: I've decided to mark it as True based on the documentation aspect.
'''
    cleaned = clean_llm_json_output(raw)
    parsed = json.loads(cleaned)  # this line is what used to raise JSONDecodeError

    assert parsed["is_match"] is False  # takes the FIRST block, not the rambled "corrected" one
