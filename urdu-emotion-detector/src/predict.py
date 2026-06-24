"""
predict.py — Single-sentence inference + attribution scoring

Used by the Streamlit app for real-time emotion prediction.
Can also be used in NB04 for explainability analysis.
"""

import torch
import numpy as np
from preprocess import ID_TO_LABEL

from config import MAX_LENGTH


def predict(text, model, tokenizer, id2label=None, device=None, max_length=MAX_LENGTH):
    """
    Predict emotion for a single text.

    Args:
        text (str): Input text to classify
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        id2label (dict): Mapping from label ID to label name. Defaults to
            config.ID_TO_LABEL if not provided.
        device (str): Device to run on ('cuda' or 'cpu')
        max_length (int): Maximum sequence length. Defaults to config.MAX_LENGTH
            (64) — the evidence-based value the model was actually trained on
            (NB03 Section 4). Passing 128 here would silently mismatch the
            model's trained input distribution.

    Returns:
        dict: {
            'label': str - predicted emotion label,
            'confidence': float - confidence score for predicted label,
            'all_scores': dict - all class probabilities
        }
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if id2label is None:
        id2label = ID_TO_LABEL

    model.eval()
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id = int(torch.argmax(logits, dim=-1).item())

    result = {
        'label': id2label[pred_id],
        'confidence': float(probs[pred_id]),
        'all_scores': {id2label[i]: float(probs[i]) for i in range(len(probs))}
    }

    return result


def predict_batch(texts, model, tokenizer, id2label=None, device=None,
                  batch_size=32, max_length=MAX_LENGTH):
    """
    Predict emotions for a batch of texts.

    More efficient than repeated single predictions.

    Args:
        max_length (int): Defaults to config.MAX_LENGTH (64), matching what
            the model was trained on. See predict() docstring for why this
            matters — do not override to 128.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if id2label is None:
        id2label = ID_TO_LABEL

    model.eval()
    model.to(device)

    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        for j in range(len(batch_texts)):
            result = {
                'label': id2label[int(preds[j])],
                'confidence': float(probs[j][preds[j]]),
                'all_scores': {id2label[k]: float(probs[j][k]) for k in range(len(probs[j]))}
            }
            all_results.append(result)

    return all_results


def get_attribution_scores(text, model, tokenizer, device=None):
    """
    Get token-level attribution scores for explainability.

    Uses transformers-interpret library for SHAP-based attribution.
    Requires: pip install "transformers==4.40.0" "transformers-interpret==0.10.0"

    Note: newer transformers versions (5.x) break transformers-interpret's
    .visualize()/internal attribute access — see NB04 troubleshooting. Pin
    the versions above in whatever environment calls this function.
    """
    try:
        from transformers_interpret import SequenceClassificationExplainer
    except ImportError:
        raise ImportError(
            "transformers-interpret not installed. "
            "Install with: pip install \"transformers==4.40.0\" \"transformers-interpret==0.10.0\""
        )

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    explainer = SequenceClassificationExplainer(model, tokenizer)

    # transformers_interpret only special-cases model_type == 'roberta' when deciding
    # whether to pass explicit position_ids. XLM-RoBERTa's config.model_type is
    # 'xlm-roberta', so that check misses it, and the library sends raw
    # torch.arange(seq_len) position ids instead of RoBERTa's true padding-offset
    # position ids — this can index outside the position embedding table and raise
    # IndexError: index out of range in self. Forcing both flags off makes the model
    # compute its own (correct) position/token-type ids internally, as it does at
    # inference time.
    explainer.accepts_position_ids = False
    explainer.accepts_token_type_ids = False

    word_attributions = explainer(text)

    tokens = [item[0] for item in word_attributions]
    attributions = [item[1] for item in word_attributions]

    prediction = predict(text, model, tokenizer, device=device)

    return {
        'tokens': tokens,
        'attributions': attributions,
        'predicted_label': prediction['label'],
        'confidence': prediction['confidence']
    }
