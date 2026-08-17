"""Filters and helpers shared by the Grad-CAM sweep (book, sections 4.3 / 5.1.4).

This module used to re-declare the audio parameters and to `torch.load` a
checkpoint at import time, so importing it from anywhere required the .pt file
to sit in the current directory. The constants now come from PreprocessParams
(one source of truth) and the model is loaded on demand via `load_model`.
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from PreprocessParams import RAVDESS_LABELS, index_to_label

# Filter options.
# NOTE: INTENSITY_TO_INCLUDE = ['02'] keeps only the "strong" takes. RAVDESS
# records `neutral` at normal intensity only, so neutral is excluded from the
# Grad-CAM sweep by construction. Set it to ['01', '02'] to include neutral.
EMOTIONS_TO_INCLUDE = ['01', '02', '03', '04', '05', '06', '07', '08']
ACTORS_TO_INCLUDE = [f"{i:02d}" for i in range(1, 25)]
STATEMENTS_TO_INCLUDE = ['01', '02']
REPETITION_TO_INCLUDE = ['01', '02']
INTENSITY_TO_INCLUDE = ['02']

# Book, section 5.1.4: the masked view keeps the most important features of the
# Grad-CAM heat map. The text describes masking at 85%.
GRADCAM_MASK_QUANTILE = 0.85

# Book, section 5.1.4: Grad-CAM was run on samples the model classified with
# ~99% SoftMax confidence.
GRADCAM_MIN_CONFIDENCE = 0.99

index_emotion_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised',
}

# Model output index -> label name. Derived from the canonical label list so it
# always matches what the LabelEncoder produced during training.
label_emotion_mapping = index_to_label(RAVDESS_LABELS)


def wav_indexer(file_name: Path) -> Tuple[str, str]:
    numbers = re.findall(r'\d+', file_name.name.__str__())
    emotion_index = numbers[2]
    actor_number = numbers[-1]
    emotion = index_emotion_mapping[emotion_index]
    return emotion, actor_number

def is_valid_ravdess_file(path: Path) -> bool:
    if "_" in path.stem:
        return False  # augmented file

    parts = path.stem.split("-")
    if len(parts) != 7:
        return False  # malformed filename

    emotion, intensity, statement, repetition, actor = parts[2], parts[3], parts[4], parts[5], parts[6]
    return (
        emotion in EMOTIONS_TO_INCLUDE and
        intensity in INTENSITY_TO_INCLUDE and
        statement in STATEMENTS_TO_INCLUDE and
        repetition in REPETITION_TO_INCLUDE and
        actor in ACTORS_TO_INCLUDE
    )

def build_correlation_kernel(freq_bins = 20, time_bins = 40) -> np.ndarray:
    # Create smooth frequency profile: almost flat, small gentle slope
    freq_profile = np.linspace(1, 0.9, freq_bins)[:, np.newaxis]  # very gentle high→low

    # Smooth time modulation: soft sine wave
    time_profile = np.sin(np.linspace(0, np.pi, time_bins))[np.newaxis, :]

    # Combine profiles to get 2D kernel
    kernel = freq_profile * time_profile  # element-wise multiplication

    # Normalize: zero-mean and unit-norm
    kernel -= kernel.mean()
    kernel /= np.linalg.norm(kernel) + 1e-12

    return kernel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path, device: Optional[torch.device] = None):
    """Load a trained classifier for Grad-CAM, in eval mode, on `device`."""
    device = device or get_device()
    model = torch.load(Path(model_path), map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model
