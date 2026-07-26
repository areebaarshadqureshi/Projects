"""
train.py — 3-phase training loop with weighted loss and macro F1 tracking

Implements the training strategy from NB03:
- Phase 1: Freeze base, train head only (2 epochs, LR 2e-3)
- Phase 2: Unfreeze top 2 layers (2 epochs, LR 2e-4)
- Phase 3: Full unfreeze (2 epochs, LR 1e-5, linear warmup scheduler)

All three phases use early stopping (patience-based, monitoring val macro F1)
and only ever save the single best checkpoint per phase — matching the
notebook's stated convention ("only the single best checkpoint is ever
written to disk").
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import numpy as np


def compute_class_weights(labels, num_classes=6):
    """
    Compute class weights for weighted cross-entropy loss.

    Args:
        labels (array-like): Training labels — must already be integer
            label_id values (0..num_classes-1), not string labels. Does
            NOT need to contain every class — any class with zero samples
            in `labels` is given a default weight of 1.0 (with a warning),
            rather than raising.
        num_classes (int): Number of classes

    Returns:
        torch.Tensor: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    # IMPORTANT: only request weights for classes that actually appear in
    # `labels` here. sklearn's compute_class_weight raises ValueError if any
    # class passed in `classes=` is absent from `y=`, and with fear/surprise
    # at ~1.1% of the data each, a CV fold, filtered subset, or small debug
    # run can easily end up missing one entirely. Anything missing falls
    # back to weight 1.0 below instead of crashing.
    unique_labels = np.unique(labels)
    weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )

    missing = set(range(num_classes)) - set(unique_labels)
    if missing:
        print(f"WARNING: classes {sorted(missing)} have zero samples in this "
              f"split — defaulting their weight to 1.0. This usually means "
              f"your split is too small or not stratified.")

    # Ensure all classes have weights
    class_weights = torch.ones(num_classes)
    for i, label in enumerate(unique_labels):
        class_weights[label] = weights[i]

    return class_weights


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on validation/test set.

    Returns:
        dict: {'accuracy', 'macro_precision', 'macro_recall', 'macro_f1'}
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1
    }


def train_epoch(model, dataloader, optimizer, criterion, device, phase_name="Training", scheduler=None):
    """
    Train model for one epoch.

    Args:
        scheduler: optional LR scheduler with .step() called after each
            optimizer step. Only used in Phase 3 (linear warmup), matching
            NB03 Section 11 — Phase 1 and Phase 2 do not use a scheduler.
    """
    model.train()

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"{phase_name}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if criterion is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def _run_phase_with_early_stopping(
    model, train_loader, val_loader, optimizer, criterion, device,
    phase_label, max_epochs, patience, save_path, best_f1_so_far,
    history_bucket, scheduler=None, verbose=True
):
    """
    Shared early-stopping loop used by all 3 phases. Monitors val macro F1,
    saves the model only when it improves on best_f1_so_far (the running
    best across all phases, not just this one), and stops early if patience
    is exhausted before max_epochs is reached.

    Returns the (possibly updated) best_f1_so_far.
    """
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            f"{phase_label} E{epoch+1}", scheduler=scheduler
        )
        val_metrics = evaluate_model(model, val_loader, device)

        history_bucket['train_loss'].append(train_loss)
        history_bucket['val_f1'].append(val_metrics['macro_f1'])

        if verbose:
            print(f"{phase_label} E{epoch+1}: Loss={train_loss:.4f}, Val F1={val_metrics['macro_f1']:.4f}")

        if val_metrics['macro_f1'] > best_f1_so_far:
            best_f1_so_far = val_metrics['macro_f1']
            patience_counter = 0
            model.save_pretrained(save_path)
            if verbose:
                print(f"  -> New best model saved (val_f1={best_f1_so_far:.4f})")
        else:
            patience_counter += 1
            if verbose:
                print(f"  -> No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                if verbose:
                    print(f"  -> Early stopping triggered for {phase_label}")
                break

    return best_f1_so_far


def train_three_phase(model, train_loader, val_loader, device, class_weights=None,
                      save_path='outputs/model/',
                      phase_epochs=(10, 10, 5),      # matches NB03 Sections 9/10/11
                      phase_patience=(3, 3, 2),       # matches NB03 Sections 9/10/11
                      verbose=True):
    """
    Execute 3-phase training strategy with early stopping in every phase
    (monitoring val macro F1, patience-based) and a linear-warmup LR
    scheduler in Phase 3 — matching NB03's actual training notebook.

    Args:
        patience (int): epochs with no val F1 improvement before stopping
            a phase early. Default 2, matching NB03's Phase 1/3 convention.

    Returns:
        dict: Training history with metrics per phase
    """
    from model import freeze_base_model, unfreeze_top_layers, unfreeze_all

    history = {
        'phase1': {'train_loss': [], 'val_f1': []},
        'phase2': {'train_loss': [], 'val_f1': []},
        'phase3': {'train_loss': [], 'val_f1': []}
    }

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) if class_weights is not None else None
    best_f1 = 0.0

    # Phase 1: Head only
    if verbose:
        print("PHASE 1: Training head only (base frozen)")
    freeze_base_model(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)
    best_f1 = _run_phase_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion, device,
        phase_label="P1", max_epochs=phase_epochs[0], patience=phase_patience[0],
        save_path=save_path, best_f1_so_far=best_f1,
        history_bucket=history['phase1'], verbose=verbose
    )
    # Phase 2: Top 2 layers
    if verbose:
        print("PHASE 2: Unfreezing top 2 layers")
    unfreeze_top_layers(model, num_layers=2)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    best_f1 = _run_phase_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion, device,
        phase_label="P2", max_epochs=phase_epochs[1], patience=phase_patience[1], save_path=save_path,
        best_f1_so_far=best_f1, history_bucket=history['phase2'], verbose=verbose
    )

    # Phase 3: Full model — with linear warmup scheduler (matches NB03 Section 11)
    if verbose:
        print("PHASE 3: Full model unfrozen")
    unfreeze_all(model)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    phase3_epochs = 2
    total_steps = len(train_loader) * phase3_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = _run_phase_with_early_stopping(
        model, train_loader, val_loader, optimizer, criterion, device,
        phase_label="P3", max_epochs=phase_epochs[2], patience=phase_patience[2], save_path=save_path,
        best_f1_so_far=best_f1, history_bucket=history['phase3'], scheduler=scheduler, verbose=verbose
    )

    if verbose:
        print(f"TRAINING COMPLETE - Best F1: {best_f1:.4f}")
        print(f"Reloading best checkpoint from {save_path} before returning")

    best_state = type(model).from_pretrained(save_path).state_dict()
    model.load_state_dict(best_state)
    model.to(device)
    return history
