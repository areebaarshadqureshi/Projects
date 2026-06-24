"""
dataset.py — PyTorch Dataset class for emotion classification

Tokenizes text and returns input_ids + attention_mask + labels.
Used in NB03 training loop.
"""

import torch
from torch.utils.data import Dataset

from config import MAX_LENGTH


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for emotion classification.

    Handles tokenization and preparation of text-label pairs for training.

    Args:
        df (pd.DataFrame): DataFrame with 'text_clean' and 'label_id' columns
        tokenizer: HuggingFace tokenizer (e.g., XLMRobertaTokenizer)
        max_length (int): Maximum sequence length for tokenization.
            Defaults to config.MAX_LENGTH (64) — the evidence-based value
            measured in NB03 Section 4. Do not pass 128 here; the saved
            model was trained on sequences truncated/padded to MAX_LENGTH.

    Returns:
        dict: {
            'input_ids': tensor of shape (max_length,),
            'attention_mask': tensor of shape (max_length,),
            'labels': tensor of shape ()
        }
    """

    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.texts = df['text_clean'].tolist()
        self.labels = df['label_id'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"texts ({len(self.texts)}) and labels ({len(self.labels)}) "
                f"length mismatch in input DataFrame"
            )
        bad_labels = [l for l in self.labels if not (0 <= l < 6)]
        if bad_labels:
            raise ValueError(f"label_id values out of range [0,6): {set(bad_labels)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(train_df, val_df, tokenizer, batch_size=16, max_length=MAX_LENGTH):
    """
    Create train and validation DataLoaders.

    Args:
        train_df (pd.DataFrame): Training dataframe with 'text_clean' and 'label_id'
        val_df (pd.DataFrame): Validation dataframe
        tokenizer: HuggingFace tokenizer
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length. Defaults to config.MAX_LENGTH (64).

    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = EmotionDataset(train_df, tokenizer, max_length)
    val_dataset = EmotionDataset(val_df, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
