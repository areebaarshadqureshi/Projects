"""
preprocess.py — Text cleaning and label mapping utilities

Shared by notebooks and the Streamlit app.
Reused across NB01, NB02, NB04, NB05 and app/app.py.

clean_text() logic verified identical to NB01's real implementation —
unchanged here. Label maps import from config.py so there is a single
source of truth instead of redefining LABEL_TO_ID/ID_TO_LABEL separately
from model.py and predict.py.
"""

import re

from config import LABEL2ID as LABEL_TO_ID
from config import ID2LABEL as ID_TO_LABEL

# Kept for backwards compatibility with any code importing LABEL_MAP directly.
LABEL_MAP = {label: label for label in LABEL_TO_ID}


def clean_text(text: str) -> str:
    """
    Clean raw text while preserving emotion-relevant signals.

    Removes:
    - URLs (http/https links)
    - @mentions
    - HTML tags
    - Hashtag # symbol (keeps the word)
    - Excess whitespace
    - Elongated LETTER repeats (3+ -> 2), e.g. "bohaaat" -> "bohaat"

    Preserves:
    - Emojis (strong emotion signal) — including repeated emoji (😢😢😢),
      which is itself an intensity signal and must NOT be collapsed
    - Non-ASCII characters (Roman Urdu, accents)
    - Original casing (capitalization can indicate emphasis)
    - Punctuation (!!!, ???, etc. — carries emotion) — including repeated
      punctuation, which must NOT be collapsed for the same reason as emoji

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text

    Example:
        >>> clean_text("yaar this is SO bad https://example.com")
        'yaar this is SO bad'
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Collapse elongated LETTERS only (3+ repeats -> 2), e.g. "bohaaat" ->
    # "bohaat". Restricted to [a-zA-Z] on purpose: using a bare "." here
    # would also collapse repeated punctuation ("!!!" -> "!!") and repeated
    # emoji ("😢😢😢" -> "😢😢"), both of which carry real emotion-intensity
    # signal per the docstring above and must be left alone.
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)

    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_label(label: str) -> str:
    """
    Normalize a label string to lowercase and strip whitespace.

    Args:
        label (str): Raw label from dataset

    Returns:
        str: Normalized label

    Raises:
        ValueError: If label is not in LABEL_MAP
    """
    label = str(label).strip().lower()
    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}. Expected one of {list(LABEL_MAP.keys())}")
    return LABEL_MAP[label]


def label_to_id(label: str) -> int:
    """
    Convert label string to integer ID for model training.

    Args:
        label (str): Normalized label ('neutral', 'anger', etc.)

    Returns:
        int: Label ID (0-5)
    """
    return LABEL_TO_ID[label]


def id_to_label(label_id: int) -> str:
    """
    Convert integer ID back to label string for inference.

    Args:
        label_id (int): Label ID (0-5)

    Returns:
        str: Label string ('neutral', 'anger', etc.)
    """
    return ID_TO_LABEL[label_id]
