
from typing import Any, Callable, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import torch
import tqdm
from ConstPaths import TessPaths
from PreprocessParams import LABEL_STRINGS, MAX_SPECTOGRAM_DURATION_IN_SECONDS
from Preprocess import audio_to_mel_spectrogram
from audio_dataset import AllRawData, CREMARawData, EmotionSpecDataset, RavdessRawData, TESSRawData
from models import ResNetWithAttention
from pprint import pprint
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
    
    attr["path"] = path
    attr["true label"] = true_label
    
    # preper data to the model
    tensor_mel_spec = preproccess_like_in_dataloader(path)
    # insert data to model and get probabilities
    probabilities = get_sample_probabilities_of_model(model, tensor_mel_spec)
    
    # get the de/encoding from classes(nums 0-7 for model) to labels(strings)
    label_encoder = LabelEncoder()
    label_encoder.fit(idx2label_array) # order Doesn't matter!

    # insert all the probablities as such: "class label" : <probability_for_that_class>
    # Add probabilities for each class to attributes
    for i, prob in enumerate(probabilities):
        class_label = label_encoder.classes_[i]
        attr[f'prob {class_label}'] = round(float(prob), float_precision)  # Convert numpy float to Python float for better serialization

    # Add predicted class and its probability
    predicted_class_idx = probabilities.argmax()
    predicted_class_label = label_encoder.classes_[predicted_class_idx]
    attr["predicted label"] = predicted_class_label
    attr["predicted probability"] = round(float(probabilities[predicted_class_idx]), float_precision)

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

def tests1():
    ######### Probability Vector Dataframe #########

    # raw data loading:
    ravdess_raw_data = RavdessRawData(include_calm=True, include_aug=False)

    ravdess_raw_data.print_all_label_counts()

    all_data = AllRawData((ravdess_raw_data, ))

    train_set, val_set = all_data.train_val_test_split(0.2)

    # pytorch's dataset creation:

    train_ds = EmotionSpecDataset(train_set)
    val_ds = EmotionSpecDataset(val_set)



    # create the model:

    model = ResNetWithAttention(num_classes=8)

    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("Eli ResNetWithAttention.pt", map_location=device)
    
    """
    Get the probabilities from the model for the given dataset.
    """
    # probabilities = get_sample_probabilities_of_model(model, train_ds[0])  # Get probabilities for the first sample in the training dataset

    # Use a subset of the training dataset for quick testing
    subset_size = 10000  # Adjust this number based on how small a sample you want
    ds_subset = torch.utils.data.Subset(val_ds, range(subset_size))

    print(f"Computing probabilities for {subset_size} samples instead of {len(val_ds)} total samples")
    #// probabilities_df = get_dataset_probabilities_of_model(model, ds_subset)
    # print(f"True label: {label}")
    # print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities_df}")
    
    # Save the DataFrame to CSV
    filepath = "probabilities.csv"
    probabilities_df.to_csv(filepath, index=False)
    print(f"Saved probabilities to {filepath}")

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
    all_probs = []

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
            #all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    #all_probs = torch.cat(all_probs, dim=0)

    return all_preds #, all_probs

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

def test2():
    raw_data = RavdessRawData(include_calm=True, include_aug=False)
    
    all_data = AllRawData((raw_data, ), val_ratio=0.3)

    all_data = list(all_data.all_data) # must turn to list before passing to dataset to have same order of samples and predictions
    
    dataset = EmotionSpecDataset(all_data)

    model = ResNetWithAttention(num_classes=8)
    
    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load state dict first (always map to device at this point)
    model = torch.load("ResNetWithAttention.pt", map_location=device)
    model.to(device)

    # Get predictions
    preds = predict(model, dataset)
    print(all_data[:4])  # Print first 4 samples to verify alignment
    print(preds[:4])  # Print first 4 predictions
    
if __name__ == "__main__":
    ######### Probability Vector Dataframe #########

    #? 1. choose dataset
    # raw data loading:
    crema_raw_data = CREMARawData()

    data = set(list(crema_raw_data.all_data))
    
    #? 2. choose the trained model
    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(r"CREMA-D\models\2025-09-29_14-16-05\ResNetWithAttention.pt", map_location=device)
    
    #? 3. define the label2idx_array of the dataset (the order matters) 
    label2idx_array = [ 
                        LABEL_STRINGS.ANGRY,
                        LABEL_STRINGS.DISGUSTED,
                        LABEL_STRINGS.FEARFUL,
                        LABEL_STRINGS.HAPPY,
                        LABEL_STRINGS.NEUTRAL,
                        LABEL_STRINGS.SAD
                      ]
    
    #? 4. Get the attributes for the sample
    attributes = get_raw_dataset_attributes(model, data, label2idx_array)
    
    #? 5. choose result's path (csv)
    # save the df to csv
    try:
        attributes.to_csv(r"CREMA-D/prob_vector_tables/with test cremaD speaker shuffled.csv", index=False)
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        input("ensure file is closed and press Enter to continue...")
        attributes.to_csv(r"CREMA-D/prob_vector_tables/with test cremaD speaker shuffled.csv", index=False)
    
    print("Saved to with test cremaD speaker shuffled.csv")

