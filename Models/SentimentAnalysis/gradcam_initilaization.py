# --------------------------------------------------------------
# 0.  Imports – add only TWO lines
# --------------------------------------------------------------
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from PreprocessParams import MAX_SPECTOGRAM_DURATION_IN_SECONDS

# Audio params
FREQUENCY_BIN_COUNT = 64
SAMPLE_RATE = 16000
N_FFT = 512
WINDOW_LENGTH = N_FFT
HOP_LENGTH = N_FFT // 2
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)
TARGET_FRAMES = (MAX_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1

# Filter options
EMOTIONS_TO_INCLUDE = ['01', '02', '03', '04', '05', '06', '07', '08']
ACTORS_TO_INCLUDE = [f"{i:02d}" for i in range(1, 25)]
STATEMENTS_TO_INCLUDE = ['01', '02']
REPETITION_TO_INCLUDE = ['01', '02']
INTENSITY_TO_INCLUDE = ['02']

index_emotion_mapping = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
}

label_emotion_mapping = {
    0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
    4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised',
}

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

# Left for hand_picking only!

RECORDINGS_TO_PROCESS_HANDPICKED = []

# For automated picking!
RECORDINGS_TO_PROCESS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(Path("ResNetWithAttention_70-10-20.pt"), map_location=device, weights_only=False)
model.eval()


## FOR TESTS:
if __name__ == "__main__":
    pass
