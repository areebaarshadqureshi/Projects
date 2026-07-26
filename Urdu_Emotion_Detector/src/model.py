"""
model.py — XLM-RoBERTa wrapper with classification head

Loads xlm-roberta-base from HuggingFace Hub and adds a linear classification
head for 6-class emotion prediction.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_LENGTH, NUM_LABELS
from preprocess import ID_TO_LABEL, LABEL_TO_ID
from predict import predict as predict_fn


class EmotionClassifier:
    """
    Wrapper for XLM-RoBERTa emotion classification model.

    Handles model loading, prediction, and pushing to HuggingFace Hub.
    """

    def __init__(self, model_name='xlm-roberta-base', num_labels=NUM_LABELS, device=None):
        """
        Initialize the emotion classifier.

        Args:
            model_name (str): HuggingFace model name or path
            num_labels (int): Number of emotion classes (default: 6, from config.NUM_LABELS)
            device (str): Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels

        # Label mappings — single source of truth in config.py
        self.id2label = ID_TO_LABEL
        self.label2id = LABEL_TO_ID

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, text, return_probs=True):
        """
        Predict emotion for a single text.

        Delegates to predict.predict() rather than re-implementing
        tokenize -> forward -> softmax -> argmax here. There used to be a
        second copy of that logic in this method, and it had drifted from
        predict.py's fixed version (missing `dim=-1` on torch.argmax — a
        bug that only showed up at batch size > 1). Keeping one
        implementation means that class of bug can't reappear by editing
        only one of the two copies.

        Args:
            text (str): Input text
            return_probs (bool): Whether to return all class probabilities

        Returns:
            dict: {
                'label': predicted emotion label,
                'confidence': confidence score for predicted label,
                'all_scores': dict of all class probabilities (if return_probs=True)
            }
        """
        result = predict_fn(
            text, self.model, self.tokenizer,
            id2label=self.id2label, device=self.device
        )
        if not return_probs:
            result.pop('all_scores', None)
        return result

    def save(self, save_path):
        """Save model and tokenizer to local directory."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def push_to_hub(self, repo_name, commit_message="Upload emotion classification model"):
        """Push model and tokenizer to HuggingFace Hub."""
        self.model.push_to_hub(repo_name, commit_message=commit_message)
        self.tokenizer.push_to_hub(repo_name, commit_message=commit_message)
        print(f"Model pushed to hub: {repo_name}")

    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """
        Load a fine-tuned model from local path or HuggingFace Hub.

        Args:
            model_path (str): Path to saved model or HuggingFace repo name
            device (str): Device to run on

        Returns:
            EmotionClassifier: Loaded model instance
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Create instance
        instance = cls.__new__(cls)
        instance.device = device
        instance.model = model.to(device)
        instance.tokenizer = tokenizer
        instance.num_labels = model.config.num_labels
        instance.id2label = model.config.id2label
        instance.label2id = model.config.label2id

        return instance


def freeze_base_model(model):
    """
    Freeze all base XLM-RoBERTa layers (for Phase 1 training).

    Args:
        model: HuggingFace model with roberta attribute
    """
    for param in model.roberta.parameters():
        param.requires_grad = False


def unfreeze_top_layers(model, num_layers=2):
    """
    Unfreeze top N transformer layers + pooler (for Phase 2 training).

    Args:
        model: HuggingFace model with roberta.encoder.layer attribute
        num_layers (int): Number of top layers to unfreeze
    """
    # Keep base frozen first
    freeze_base_model(model)

    # Unfreeze top N layers
    for layer in model.roberta.encoder.layer[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Unfreeze pooler
    for param in model.roberta.pooler.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """
    Unfreeze all model parameters (for Phase 3 training).

    Args:
        model: HuggingFace model
    """
    for param in model.parameters():
        param.requires_grad = True
