"""Global pre-processing constants.

These values are the ones reported in the project book, section 3.2
("Pre-processing"):

    * resample every clip to 16 kHz
    * FFT size 512, window length == FFT size, hop length 256 (== n_fft / 2)
    * 64 mel frequency bins
    * pad / truncate every spectrogram to 4.5 seconds

Every other module must import these constants instead of re-declaring them,
so that the training pipeline, Grad-CAM and TCAV all see identical inputs.
"""

from typing import Callable

import numpy as np

FREQUENCY_BIN_COUNT = 64
SAMPLE_RATE = 16000

N_FFT = 512
WINDOW_LENGTH = N_FFT
HOP_LENGTH = N_FFT // 2
MAX_SPECTOGRAM_DURATION_IN_SECONDS = 4.5 #Set to 7.06 for double_sentence training, for 4.5 otherwise.

#Calculate the max number of samples for the target duration
MAX_SAMPLES = int(MAX_SPECTOGRAM_DURATION_IN_SECONDS * SAMPLE_RATE)

# Calculate the target number of frames for the spectrogram
TARGET_FRAMES = (MAX_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1 # noam: this is the right formula for the number of frames
TOP_DB = 20

# Number of stride-2 stages in ResNetWithAttention (module1..module3). Used to
# derive the flattened feature size of the first fully-connected layer.
DOWNSAMPLE_STAGES = 3


def downsampled_size(size: int, stages: int = DOWNSAMPLE_STAGES) -> int:
    """Spatial size after `stages` stride-2 3x3 convolutions with padding 1.

    A stride-2, kernel-3, padding-1 convolution maps n -> floor((n - 1) / 2) + 1,
    which equals ceil(n / 2). Chaining that is NOT the same as `size // 2**stages`
    unless `size` happens to be divisible by 2**stages, so the FC input size must
    be derived with this helper rather than with an integer division.
    """
    for _ in range(stages):
        size = (size + 1) // 2
    return size


# --------------------------------------------------------------------------
# Spectrogram value normalisation
# --------------------------------------------------------------------------
# The book states the spectrogram is "normalized with volume standardization
# [...] after the entire conversion process". That volume normalisation is
# performed by `librosa.power_to_db(..., ref=np.max)` inside
# `Preprocess.audio_to_mel_spectrogram`, which puts every recording's peak at
# 0 dB regardless of how loud it was recorded.
#
# `SPECTROGRAM_NORMALIZATION` is an *additional*, optional transform applied on
# top of that. It is identity by default, which is the behaviour every existing
# checkpoint in this project was trained with. Swap it for
# `Preprocess.standardization` (zero mean / unit variance) only if you intend to
# retrain everything - mixing the two silently invalidates a trained model.
def identity_normalization(spectrogram: np.ndarray) -> np.ndarray:
    """No-op normalisation (the peak has already been referenced to 0 dB)."""
    return spectrogram


SPECTROGRAM_NORMALIZATION: Callable[[np.ndarray], np.ndarray] = identity_normalization


# --------------------------------------------------------------------------
# TCAV concept patches -> model input scale
# --------------------------------------------------------------------------
# `librosa.power_to_db(..., ref=np.max)` with the default top_db=80 produces
# values in [-80, 0] dB, so that is the range every real input to the model
# occupies. The synthetic concept patches in `concepts_creation` are normalised
# to [0, 1] instead.
#
# That mismatch matters: the network's BatchNorm layers are calibrated for
# dB-scale inputs, so a [0, 1] patch lands roughly +3 sigma away from anything
# the model was trained on and the concept/random activations collapse together.
# Measured on a trained classifier, a linear probe separating one concept from
# the random baseline scores 0.50 (chance) on [0, 1] patches and 1.00 on the
# same patches mapped to dB - i.e. the CAVs are otherwise fitted to noise.
#
# The book states the concepts "appear in the spectrogram format, corresponding
# to the input data format and usable by the CNN" (section 4.4), which is what
# this mapping actually delivers.
CONCEPT_DB_FLOOR = -80.0


def patch_to_db_scale(patch: np.ndarray, db_floor: float = CONCEPT_DB_FLOOR) -> np.ndarray:
    """Map a [0, 1] concept patch onto the [db_floor, 0] dB range of real inputs.

    Must be applied to the positive concepts *and* the random baseline, or the
    linear classifier separates them on scale alone and the CAV is meaningless.
    """
    return patch * (-db_floor) + db_floor


class LABEL_STRINGS:
    """Canonical emotion label spellings used across every dataset adapter."""

    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CALM = "calm"


# The label order the model's output indices follow. `sklearn`'s LabelEncoder
# sorts classes alphabetically, so any mapping from a model index back to a
# label name must use the sorted subset of the labels present in that dataset.
RAVDESS_LABELS = sorted([
    LABEL_STRINGS.ANGRY, LABEL_STRINGS.CALM, LABEL_STRINGS.DISGUSTED,
    LABEL_STRINGS.FEARFUL, LABEL_STRINGS.HAPPY, LABEL_STRINGS.NEUTRAL,
    LABEL_STRINGS.SAD, LABEL_STRINGS.SURPRISED,
])

TESS_LABELS = sorted([
    LABEL_STRINGS.ANGRY, LABEL_STRINGS.DISGUSTED, LABEL_STRINGS.FEARFUL,
    LABEL_STRINGS.HAPPY, LABEL_STRINGS.NEUTRAL, LABEL_STRINGS.SAD,
    LABEL_STRINGS.SURPRISED,
])

CREMAD_LABELS = sorted([
    LABEL_STRINGS.ANGRY, LABEL_STRINGS.DISGUSTED, LABEL_STRINGS.FEARFUL,
    LABEL_STRINGS.HAPPY, LABEL_STRINGS.NEUTRAL, LABEL_STRINGS.SAD,
])


def label_to_index(labels: list[str]) -> dict[str, int]:
    """Map every label name to the output index the model uses for it."""
    return {label: idx for idx, label in enumerate(sorted(labels))}


def index_to_label(labels: list[str]) -> dict[int, str]:
    """Inverse of :func:`label_to_index`."""
    return {idx: label for idx, label in enumerate(sorted(labels))}
