"""Per-sample class probabilities for a trained classifier.

Produces the "probability vector" table that `tcav_demo.py` consumes and that
the Grad-CAM sweep filters on (book, section 5.1.4: only samples the model
classified with ~99% SoftMax confidence were visualised).

    python prob_vector.py ravdess --model <model.pt> --out ravdess_prob_vector.csv
    python prob_vector.py tess    --model <model.pt> --out tess_prob_vector.csv
    python prob_vector.py cremad  --model <model.pt> --out cremad_prob_vector.csv

Columns: path, true_label, prob_<class> for every class, predicted_label,
predicted_probability. Underscore column names match what tcav_demo.py and the
analysis notebooks expect.
"""

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import tqdm
from PreprocessParams import (CREMAD_LABELS, MAX_SPECTOGRAM_DURATION_IN_SECONDS,
                              RAVDESS_LABELS, TESS_LABELS)
from Preprocess import audio_to_mel_spectrogram
from audio_dataset import (CremaDSplitttedRawData, RavdessRawData,
                           TessSplitttedRawData, sorted_samples)
import torch.nn as nn
from torch.utils.data import DataLoader

def get_sample_probabilities_of_model(model, mel_spec_tensor):
    """
    Get the probabilities from the model for a tensor of a single sample.
    """ 
    model.eval()
    # Add batch dimension since model expects input in shape [B, C, H, W]
    if mel_spec_tensor.dim() == 3:
        # If input_tensor is 3D, add a batch dimension
        mel_spec_tensor = mel_spec_tensor.unsqueeze(0)
    
    # Move input and model to the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mel_spec_tensor = mel_spec_tensor.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        logits = model(mel_spec_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        # predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # turn from 2d to 1d
    probabilities = probabilities.squeeze(0)  # Remove batch dimension if present
    return probabilities.cpu().numpy()

def get_sample_result_of_model(model, mel_spec_tensor):
    probabilities = get_sample_probabilities_of_model(model, mel_spec_tensor)
    predicted_class = probabilities.argmax()
    return predicted_class

def preproccess_like_in_dataloader(file_path):
    # noam: audio_to_mel_spectrogram returns shape (freq_bins, time_frames)
    mel_spectrogram = audio_to_mel_spectrogram(file_path=file_path, max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS)

    # Convert to torch.Tensor
    mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
    
    # Now expand to shape (1, freq_bins, time_frames)
    mel_spectrogram = mel_spectrogram.unsqueeze(dim=0)
    
    return mel_spectrogram

def get_raw_sample_attributes(model, raw_data_sample, idx2label_array, float_precision=3):
    """
    Get the probabilities from the model for a single sample from audio raw data sample
    
    Args:
        model: The trained model to use for prediction.
        raw_data_sample: (Path, label) tuple where Path is the path to the audio file and label is the emotion label.
    Returns:
        attributes: A dictionary containing the predicted class and its probability.  
            probabilities of each label
            predicted label 
            predicted probability 
            true label 
            path 
    """
    attr = {} # dictionary to hold attributes
    path, true_label = raw_data_sample

    attr["path"] = str(path)
    attr["true_label"] = true_label

    # preper data to the model
    tensor_mel_spec = preproccess_like_in_dataloader(path)
    # insert data to model and get probabilities
    probabilities = get_sample_probabilities_of_model(model, tensor_mel_spec)

    # Model output index i corresponds to the i-th class in *sorted* order,
    # because EmotionSpecDataset encodes labels with sklearn's LabelEncoder.
    class_names = sorted(idx2label_array)
    if len(class_names) != len(probabilities):
        raise ValueError(
            f"Model produced {len(probabilities)} logits but {len(class_names)} class names "
            f"were declared: {class_names}"
        )

    # insert all the probablities as such: "prob_<class label>" : <probability>
    for i, prob in enumerate(probabilities):
        attr[f'prob_{class_names[i]}'] = round(float(prob), float_precision)

    # Add predicted class and its probability
    predicted_class_idx = int(probabilities.argmax())
    attr["predicted_label"] = class_names[predicted_class_idx]
    attr["predicted_probability"] = round(float(probabilities[predicted_class_idx]), float_precision)

    # Convert the dictionary to a pandas Series
    
    # Create a Series with the specified order WARNING: This will only include keys that are present in attr_order
    attr_series = pd.Series(attr)
    
    return attr_series

def get_raw_dataset_attributes(model, raw_dataset: set, idx2label_array):
    """
    Get the attributes for each sample in the raw dataset.
    
    Args:
        model: The trained model to use for prediction.
        raw_dataset: An iterable of raw data samples, where each sample is a (Path, label) tuple.
        
    Returns:
        attributes_list: A list of dictionaries, each containing attributes for a sample.
    """
    attributes_list = []
    for sample in tqdm.tqdm(raw_dataset, desc="Processing samples", unit="sample"):
        attributes = get_raw_sample_attributes(model, sample, idx2label_array)
        attributes_list.append(attributes)
    
    # Convert the list of Series objects to a DataFrame
    attributes_df = pd.DataFrame(attributes_list)
    
    return attributes_df

def predict(classification_model: nn.Module, dataset: torch.utils.data.Dataset):
    """
    Predict the classes for all samples in the dataset using the provided classification model.
    """
    # Put model in evaluation mode
    classification_model.eval()

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # shuffle=False means the order of samples is preserved

    # Device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model.to(device)

    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle case: dataset returns only input or (input, label)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _ = batch
            else:
                inputs = batch

            inputs = inputs.to(device)

            logits = classification_model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds, dim=0)

    return all_preds

def evaluate_single_label(
    model: nn.Module,
    data: torch.utils.data.Dataset,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any]]] = None,
) -> Dict[str, Any]:
    """
    Single-label evaluator.
    - Only accepts a Dataset (preserves original order; no shuffling).
    - No built-in metrics are computed. Anything you want (accuracy, F1, confusion, top-k, etc.)
      should be supplied via `metrics` as callables.

    metrics API:
        fn(y_true, y_pred, y_prob, logits) -> Any
        where:
            y_true:  (N,) int32/64
            y_pred:  (N,) int32/64
            y_prob:  (N, C) float32 (softmax probs)
            logits:  (N, C) float32 (raw)
    Returns:
        {
          "avg_loss": float,
          "num_samples": int,
          "custom": {name: result, ...}  # only if metrics provided
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    # Build a deterministic DataLoader internally (no batch_size/topk exposed)
    loader = DataLoader(data, batch_size=64, shuffle=False)

    model.eval()
    model.to(device)

    total_loss = 0.0
    n_samples = 0

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Dataset must yield (x, y).")
            x, y = batch[0], batch[1]

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)                   # (B, C)
            loss = loss_fn(logits, y)           # scalar

            num_classes = logits.size(1)  # C
            if (y >= num_classes).any():
                raise ValueError(
                    f"Found label(s) outside range [0, {num_classes-1}]. "
                    f"Max label={y.max().item()}, num_classes={num_classes}"
                )
            
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            bsz = x.size(0)
            total_loss += float(loss.item()) * bsz
            n_samples += bsz

            all_true.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    if n_samples == 0:
        return {"avg_loss": float("nan"), "num_samples": 0}

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    logits_np = np.concatenate(all_logits, axis=0)
    probs_np = np.concatenate(all_probs, axis=0)

    out = {
        "avg_loss": total_loss / max(n_samples, 1),
        "num_samples": n_samples,
    }

    if metrics:
        custom = {}
        for name, fn in metrics.items():
            try:
                custom[name] = fn(y_true, y_pred, probs_np, logits_np)
            except Exception as e:
                custom[name] = {"error": str(e)}
        out["custom"] = custom

    return out

def load_raw_samples(dataset: str) -> tuple[list, Sequence[str]]:
    """Every (path, label) sample of a dataset plus its declared label space."""
    if dataset == "ravdess":
        return sorted_samples(RavdessRawData(include_calm=True).all_data), RAVDESS_LABELS
    if dataset == "tess":
        return sorted_samples(TessSplitttedRawData().all_data), TESS_LABELS
    if dataset == "cremad":
        return sorted_samples(CremaDSplitttedRawData().all_data), CREMAD_LABELS
    raise ValueError(f"Unknown dataset {dataset!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["ravdess", "tess", "cremad"])
    parser.add_argument("--model", type=Path, required=True, help="trained .pt classifier")
    parser.add_argument("--out", type=Path, required=True, help="destination CSV")
    args = parser.parse_args()

    samples, label_space = load_raw_samples(args.dataset)
    print(f"{len(samples)} samples, {len(label_space)} classes: {sorted(label_space)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = torch.load(args.model, map_location=device, weights_only=False)

    attributes = get_raw_dataset_attributes(trained_model, samples, list(label_space))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    attributes.to_csv(args.out, index=False)
    print(f"wrote {len(attributes)} rows to {args.out}")

