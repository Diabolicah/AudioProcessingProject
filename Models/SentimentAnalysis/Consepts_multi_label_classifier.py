"""
    define: 
        dataset; train, evaluate and predict methods
    
    execute:
        load model and dataset, train the model, evaluate it and save the best model.

    ALL FOR MULTI-LABEL CONCEPT CLASSIFICATION OF SPECTROGRAMS
"""


# =========================
# Multi-label spectrogram tagger (ResNet-18)
# =========================

# Imports
import os
from pathlib import Path
from pprint import pprint
from typing import Any, List, Tuple, Dict, Sequence, Optional
from dataclasses import dataclass
from xml.parsers.expat import model

from arrow import get
import pandas as pd
from regex import D
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
from collections import defaultdict

from Preprocess import audio_to_mel_spectrogram
from preproccess_for_concept_classification import prepare_spectrogram_for_backbone
from tcav_demo import CONCEPT_UNIQUE_NAMES

# ---- your function is assumed to be defined/imported beforehand ----
# from your_module import prepare_spectrogram_for_backbone

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 25
NUM_WORKERS = 0 # no multiprocessing
FREEZE_TO_LAST_BLOCK = True         # freeze all except layer4 + fc (few-shot friendly)
CALIBRATE_THRESHOLDS = True         # find per-label thresholds on val set
CKPT_PATH = "best_resnet18_multilabel.pt"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- Your label space ----------
# List every possible tag (order defines the output indices)
ALL_TAGS: List[str] = CONCEPT_UNIQUE_NAMES
TAG2IDX = {t: i for i, t in enumerate(ALL_TAGS)}
NUM_TAGS = len(ALL_TAGS)

# ---------- Your data ----------
def get_train_val_split(file_path: str = 'spectrograms_multi_labels.csv', random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Load the CSV and split into train/val ensuring no actor overlap.
        
        assumes the CSV has columns:
        - "Path": str path to the spectrogram image
        - one column per concept in ALL_TAGS with 0/1 values indicating absence/presence
        
        Returns:
        - train_items: pd.DataFrame for training
        - val_items: pd.DataFrame for validation
    """
    all_spectrogram_concept_labeled = pd.read_csv(file_path)

    # train/val split by actor IDs

    # Extract actor ID (the last number in path)
    all_spectrogram_concept_labeled["actor"] = all_spectrogram_concept_labeled["Path"].str.extract(r"Actor_(\d+)")

    # Define groups
    groups = all_spectrogram_concept_labeled["actor"]

    # Split (80% train, 20% val) ensuring no actor appears in both
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(gss.split(all_spectrogram_concept_labeled, groups=groups))

    train_items = all_spectrogram_concept_labeled.iloc[train_idx]
    val_items = all_spectrogram_concept_labeled.iloc[val_idx]
    
    return train_items, val_items


# ---------- Dataset ----------
class MultiLabelSpectrogramDataset(Dataset):
    """
    Multi-label dataset for spectrogram images.

    Each item is a tuple (image_path, tags), where:
      - image_path: str path to a spectrogram image (PNG/JPG).
      - tags: List[str] of concept names present in the sample.

    Produces for each index:
      - model_img: torch.FloatTensor (3, 224, 224), normalized and resized for the backbone
                   using prepare_spectrogram_for_backbone.
      - y: torch.FloatTensor (NUM_TAGS,), a multi-hot vector corresponding to ALL_TAGS.
      - path: str, the original image path (useful for debugging).
    """

    def __init__(self, items: pd.DataFrame, letterbox: bool = False):
        """
        Initialize the dataset.

        Parameters:
          - items: pd.DataFrame with columns ["Path", "long_constant_thick", ...] where each row is a sample and each column is a concept. ordered by the ALL_TAGS list.
          0/1 values indicate absence/presence of each concept.
          - letterbox: Placeholder flag for future letterbox handling (not used in current pipeline).
        """
        # Keep only the path and the label columns in the expected order
        cols = ["Path"] + ALL_TAGS
        missing = [c for c in cols if c not in items.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        self.items = items[cols].reset_index(drop=True).copy()
        self.letterbox = letterbox
        
        #! debug
        print(f"Dataset initialized with {self.items.shape} shape.")
        #! debug

    def __len__(self):
        """
        Returns:
          int: Number of samples in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx: int):
        """
        Build a multi-hot label vector and return the backbone-ready image.

        Returns:
          Tuple[torch.FloatTensor, torch.FloatTensor, str]:
            - model_img: (3, 224, 224) float tensor, normalized as required by the backbone.
            - y: (NUM_TAGS,) float multi-hot vector aligned with ALL_TAGS.
            - path: Original image path.

        Notes:
          - Uses prepare_spectrogram_for_backbone with out_size=(224, 224), to_rgb=True, normalize=True.
        """
        row = self.items.iloc[idx]
        path = Path(row["Path"])

        # Select labels strictly by ALL_TAGS and ensure float32
        y = torch.tensor(row[ALL_TAGS].to_numpy(dtype=np.float32), dtype=torch.float32)
        
        # load spectrogram image:
        img = audio_to_mel_spectrogram(path)
        img = torch.from_numpy(img)
        # your function handles resizing + normalization for ResNet
        # we only need the model_img return
        _, model_img = prepare_spectrogram_for_backbone(
            img,
            out_size=(224, 224),
            letterbox=False,
            to_rgb=True,
            normalize=True,
            return_both=True,
        )
        return model_img, y, str(path)  # keep path for debugging, sending as path will make dataloader create an error.

# ---------- Dummy baseline model: always predict 1 (exists) for every tag ----------
class DummyAllOnes(nn.Module):
    """A dummy model that ignores the input and outputs the same large positive logit
    for every class, so that sigmoid(logit) ~ 1. Works with evaluate()."""
    def __init__(self, num_tags: int, logit_value: float = 1000.0):
        super().__init__()
        self.num_tags = num_tags
        # register a non-trainable buffer scalar; sigmoid(1000) ~ 1
        self.register_buffer("logit", torch.tensor(float(logit_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        # broadcast the scalar logit to (n, num_tags)
        return self.logit.expand(n, self.num_tags)

def get_pretrained_model():
    # ---------- Model ----------
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_dim = base.fc.in_features
    base.fc = nn.Linear(in_dim, NUM_TAGS)
    model = base.to(DEVICE)

    if FREEZE_TO_LAST_BLOCK:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True
    return model

#? ---------- Metrics ----------
def get_eval_metric_dict(y_true: torch.Tensor, y_prob: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    y_true: (N, C) float {0,1}
    y_prob: (N, C) sigmoid probabilities
    thresholds: (C,) per-label thresholds in [0,1] or None->0.5
    """
    metric_dict = {"per class": None, "global": None}
    
    eps = 1e-9
    if thresholds is None:
        thresholds = torch.full((y_true.shape[1],), 0.5, device=y_prob.device)
    y_pred = (y_prob >= thresholds.view(1, -1)).float()

    # per-class stats
    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1 - y_true)).sum(dim=0)
    fn = ((1 - y_pred) * y_true).sum(dim=0)

    metric_dict["per class"] = {"true positive": tp, "false positive": fp, "false negative": fn}
    
    precision_c = tp / (tp + fp + eps)
    recall_c    = tp / (tp + fn + eps)
    f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)
    
    metric_dict["per class"].update({"precision": precision_c, "recall": recall_c, "f1": f1_c})
    
    # macro
    f1_macro = f1_c.mean().item()
    
    metric_dict["global"] = {"f1_macro": f1_macro}
    
    # micro (sum over classes first)
    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    precision_micro = TP / (TP + FP + eps)
    recall_micro    = TP / (TP + FN + eps)
    f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + eps)).item()

    metric_dict["global"].update({"true positive micro": TP, "false positive micro": FP, "false negative micro": FN, "precision micro": precision_micro, "recall micro": recall_micro, "f1_micro": f1_micro})

    return metric_dict

def f1_micro_macro(y_true: torch.Tensor, y_prob: torch.Tensor, thresholds: Optional[torch.Tensor] = None):
    metrics = get_eval_metric_dict(y_true, y_prob, thresholds)
    f1_micro = metrics["global"]["f1_micro"]
    f1_macro = metrics["global"]["f1_macro"]
    f1_c = metrics["per class"]["f1"]
    return f1_micro, f1_macro, f1_c.detach().cpu().numpy()

def calibrate_thresholds(y_true: torch.Tensor, y_logit: torch.Tensor) -> torch.Tensor:
    """
    For each class, sweep thresholds and pick the one that maximizes F1 on the validation set.
    Returns a (C,) tensor in [0,1].
    """
    y_prob = torch.sigmoid(y_logit)
    C = y_true.shape[1]
    thresholds = torch.zeros(C, device=y_true.device)
    grid = torch.linspace(0.05, 0.95, steps=19, device=y_true.device)
    eps = 1e-9
    for c in range(C):
        best_t, best_f1 = 0.5, -1.0
        p = y_prob[:, c]
        t = y_true[:, c]
        for th in grid:
            pred = (p >= th).float()
            tp = (pred * t).sum()
            fp = (pred * (1 - t)).sum()
            fn = ((1 - pred) * t).sum()
            prec = tp / (tp + fp + eps)
            rec  = tp / (tp + fn + eps)
            f1 = (2 * prec * rec / (prec + rec + eps)).item()
            if f1 > best_f1:
                best_f1, best_t = f1, th.item()
        thresholds[c] = best_t
    return thresholds

# ---------- Train / Validate ----------
def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train for a single epoch over a multi-label spectrogram dataset.

    Args:
        model (torch.nn.Module): Model that maps a batch of images to logits of shape (N, C).
        loader (torch.utils.data.DataLoader): Yields tuples (imgs, targets, path) where:
            - imgs: Float tensor (N, 3, 224, 224) to be moved to DEVICE.
            - targets: Float multi-hot tensor (N, C) aligned with ALL_TAGS.
            - path: Original sample path (unused here).
        optimizer (torch.optim.Optimizer): Optimizer for updating trainable parameters.
        criterion (Callable): Loss function taking (logits, targets) -> scalar loss
            (e.g., nn.BCEWithLogitsLoss for multi-label classification).

    Returns:
        float: Average loss over the entire dataset (sum of batch losses normalized by dataset size).

    Notes:
        - Sets model.train().
        - Respects requires_grad filtering (useful when freezing layers).
        - Uses optimizer.zero_grad(set_to_none=True) for performance.
    """
    model.train()
    running_loss = 0.0
    for imgs, targets, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_multi_label(model, loader, criterion):
    """
    Evaluate the model on a validation/test split without gradient tracking.

    Args:
        model (torch.nn.Module): Model that produces logits of shape (N, C).
        loader (torch.utils.data.DataLoader): Yields (imgs, targets, path) batches.
        criterion (Callable): Loss function for reporting average loss.

    Returns:
        Tuple[
            float,               # avg_loss: Mean loss over the dataset
            float,               # f1_micro: Micro-averaged F1 at threshold 0.5
            float,               # f1_macro: Macro-averaged F1 at threshold 0.5
            torch.Tensor,        # logits: Concatenated raw logits of shape (N, C)
            torch.Tensor         # targets: Concatenated multi-hot targets of shape (N, C)
        ]

    Notes:
        - Sets model.eval() and runs under torch.no_grad().
        - F1 metrics are computed with f1_micro_macro using default per-class threshold 0.5
          (thresholds=None). For calibrated thresholds, pass them explicitly or adapt this call.
        - Moves batches to DEVICE with non_blocking=True.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * imgs.size(0)

        all_logits.append(logits)
        all_targets.append(targets)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits)
    metrics_dict = get_eval_metric_dict(targets, probs, thresholds=None)
    f1_micro, f1_macro, f1_per_class = metrics_dict["global"]["f1_micro"], metrics_dict["global"]["f1_macro"], metrics_dict["per class"]["f1"].detach().cpu().numpy()
    return {
        "loss": total_loss / len(loader.dataset),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "logits": logits,          # (N, C)
        "targets": targets,        # (N, C)
        "extra": {
            "f1_per_class": f1_per_class,  # np.array, optional
            "precition_per_class": metrics_dict["per class"]["precision"].detach().cpu().numpy(),
            "recall_per_class": metrics_dict["per class"]["recall"].detach().cpu().numpy()
        }
    }

# ---------- Inference helper ----------
@torch.no_grad()
def predict_paths(paths: List[str], ckpt_path: str = CKPT_PATH, letterbox: bool=False):
    model = get_pretrained_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    thresholds = ckpt.get("thresholds", torch.full((NUM_TAGS,), 0.5)).to(DEVICE)

    model.eval()
    preds = []
    for p in paths:
        # load + prepare
        img = torch.from_numpy(audio_to_mel_spectrogram(Path(p)))
        _, x_m = prepare_spectrogram_for_backbone(img, out_size=(224,224), letterbox=letterbox,
                                                  to_rgb=True, normalize=True, return_both=True)
        x_m = x_m.unsqueeze(0).to(DEVICE)

        logit = model(x_m)
        prob = torch.sigmoid(logit).squeeze(0)          # (C,)
        pred = (prob >= thresholds).nonzero(as_tuple=False).view(-1).tolist()
        tags = [ALL_TAGS[i] for i in pred]
        preds.append({"path": p, "probs": prob.cpu().numpy(), "tags": tags})
    return preds


def train_evaluate_save():
    train_items, val_items = get_train_val_split(file_path='spectrograms_multi_labels.csv', random_state=42)
    
    print(f"Train items shape: {train_items.shape}, Val items shape: {val_items.shape}")
    
    # ---------- DataLoaders ----------
    train_ds = MultiLabelSpectrogramDataset(train_items, letterbox=False)
    val_ds   = MultiLabelSpectrogramDataset(val_items,   letterbox=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = get_pretrained_model()
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # ---------- Fit ----------
    best_val_score = -1.0
    best_thresholds = torch.full((NUM_TAGS,), 0.5)

    display_f1_per_class = None
    best_metrics_dict = {}
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        metrics_dict = evaluate_multi_label(model, val_loader, criterion)
        val_loss, f1_micro, f1_macro, val_logits, val_targets, f1_per_class = metrics_dict['loss'], metrics_dict['f1_micro'], metrics_dict['f1_macro'], metrics_dict['logits'], metrics_dict['targets'], metrics_dict['extra']['f1_per_class']
        # optional per-label threshold calibration on validation set
        if CALIBRATE_THRESHOLDS:
            best_thresholds = calibrate_thresholds(val_targets, val_logits)
            val_probs = torch.sigmoid(val_logits)
            f1_micro_cal, f1_macro_cal, f1_per_class_cal = f1_micro_macro(val_targets, val_probs, thresholds=best_thresholds)
            display_f1 = f1_micro_cal
            display_macro = f1_macro_cal
            display_f1_per_class = f1_per_class_cal
        else:
            display_f1 = f1_micro
            display_macro = f1_macro
            display_f1_per_class = f1_per_class

        print(f"[{epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"F1_micro={display_f1:.4f}  F1_macro={display_macro:.4f}  ")
        

        # scheduler on micro-F1
        scheduler.step(display_f1)

        if display_f1 > best_val_score:
            best_val_score = display_f1
            best_metrics_dict = metrics_dict
            torch.save({
                "model_state": model.state_dict(),
                "thresholds": best_thresholds.cpu(),
                "all_tags": ALL_TAGS,
            }, CKPT_PATH)
            print(f"  ↪ saved best checkpoint to {CKPT_PATH} (F1_micro={best_val_score:.4f})")
    print('best metrics: ')
    print(best_metrics_dict)
    print("Training done.") 


if __name__ == "__main__":
    # model = DummyAllOnes(NUM_TAGS).to(DEVICE)

    # train_items, val_items = get_train_val_split()
    # train_ds = MultiLabelSpectrogramDataset(train_items, letterbox=False)
    # val_ds   = MultiLabelSpectrogramDataset(val_items,   letterbox=False)
    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    # val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # pprint(evaluate(model, val_loader, nn.BCEWithLogitsLoss()))
    
    train_evaluate_save()
    
    # pprint(predict_paths([r'RAVDESS\original_data\Actor_04\03-01-07-01-02-01-04.wav']))