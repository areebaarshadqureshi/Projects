"""
config.py — single source of truth for constants shared across the pipeline.

Every other src/ module and the Streamlit app should import from here rather
than re-defining MAX_LENGTH or the label mappings locally. This is what
NB03 actually measured and trained with — if you ever change it, this is
the one place to edit.
"""

# Canonical 6-class label taxonomy used across the whole pipeline
# (NB01–NB05, training, inference, the Streamlit app). Taken from the
# original preprocess.py mapping — this is now the one place it's defined.
ID2LABEL = {
    0: "neutral",
    1: "anger",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Evidence-based, not a default: NB03 Section 4 measured the real XLM-R
# subword tokenizer's 95th percentile on text_clean and got this value.
# Do not change this back to 128 — the saved model was trained on
# sequences truncated/padded to this length.
MAX_LENGTH = 64

SEED = 42

NUM_LABELS = len(ID2LABEL)
