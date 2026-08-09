"""
A lightweight collection of *basic* audio‑data‑augmentation utilities.

All transforms work on NumPy arrays. Waveforms are expected to be either

- `shape == (n_samples,)`    # mono, or
- `shape == (n_channels, n_samples)`  # multi‑channel

Spectrogram functions expect a 2‑D NumPy array `(n_freq_bins, n_frames)`.
The code purposefully avoids heavy dependencies and only relies on
NumPy & librosa (the latter is used for high‑quality time‑stretching &
pitch‑shifting).
"""
from __future__ import annotations

import random
from typing import Union

import librosa
import numpy as np

__all__ = [
    "add_background_noise",
    "time_stretch",
    "pitch_shift",
    "change_volume",
    "random_crop",
    "pad_audio",
    "time_mask"
    # "frequency_mask",
]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _to_channels_first(wave: np.ndarray) -> np.ndarray:
    """Ensure output has shape (channels, samples)."""
    if wave.ndim == 1:
        return wave[None, :]
    return wave


def _from_channels_first(wave: np.ndarray, original_ndim: int) -> np.ndarray:
    """Return to original dimensionality (mono vs. multi)."""
    if original_ndim == 1:
        return wave[0]
    return wave


# -----------------------------------------------------------------------------
# Augmentations operating on *waveforms*
# -----------------------------------------------------------------------------

def add_background_noise(
    waveform: np.ndarray,
    noise_factor: float = 0.005,
    noise_source: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Add Gaussian (or provided) noise at a given *linear* scale.

    Parameters
    ----------
    waveform: np.ndarray
        Input audio (mono or multi‑channel)
    noise_factor: float
        Linear scaling factor for the noise. Typical range 0.001‑0.02.
    noise_source: np.ndarray | None
        If provided, this array (same length as *waveform*) is used instead of
        white noise.
    """
    wave = waveform.astype(np.float32)
    if noise_source is None:
        noise = np.random.randn(*wave.shape)
    else:
        noise = noise_source.astype(np.float32)
        if noise.shape != wave.shape:
            raise ValueError("noise_source shape must match waveform shape")
    augmented = wave + noise_factor * noise
    return augmented.astype(waveform.dtype)


def time_stretch(
    waveform: np.ndarray,
    rate: float = 1.1,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Speed up (>1.0) or slow down (<1.0) without altering pitch.

    Uses librosa's phase‑vocoder time‑stretch per channel.
    """
    original_ndim = waveform.ndim
    wave_c = _to_channels_first(waveform)
    stretched_channels = []
    for ch in wave_c:
        stretched_channels.append(librosa.effects.time_stretch(ch, rate=rate))
    # Pad / trim so all channels equal length
    max_len = max(len(ch) for ch in stretched_channels)
    stretched = np.stack([
        np.pad(ch, (0, max_len - len(ch))) for ch in stretched_channels
    ])
    return _from_channels_first(stretched, original_ndim)


def pitch_shift(
    waveform: np.ndarray,
    sample_rate: int,
    n_steps: float = 2.0,
) -> np.ndarray:
    """Shift pitch by *n_steps* (in semitones) without changing tempo."""
    original_ndim = waveform.ndim
    wave_c = _to_channels_first(waveform)
    shifted_channels = [
        librosa.effects.pitch_shift(ch, sr=sample_rate, n_steps=n_steps)
        for ch in wave_c
    ]
    max_len = max(len(ch) for ch in shifted_channels)
    shifted = np.stack([
        np.pad(ch, (0, max_len - len(ch))) for ch in shifted_channels
    ])
    return _from_channels_first(shifted, original_ndim)


def change_volume(
    waveform: np.ndarray,
    gain_db: float = 3.0,
) -> np.ndarray:
    """Apply gain (± dB). Positive boosts, negative attenuates."""
    factor = 10 ** (gain_db / 20.0)
    return (waveform.astype(np.float32) * factor).astype(waveform.dtype)


def random_crop(
    waveform: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    """Return a random *crop_size* segment. Pads if audio shorter."""
    if waveform.shape[-1] >= crop_size:
        start = random.randint(0, waveform.shape[-1] - crop_size)
        if waveform.ndim == 1:
            return waveform[start : start + crop_size]
        else:
            return waveform[:, start : start + crop_size]
    else:
        # Pad to required length
        pad_len = crop_size - waveform.shape[-1]
        pad_width = ((0, 0), (0, pad_len)) if waveform.ndim == 2 else ((0, pad_len),)
        return np.pad(waveform, pad_width)


def pad_audio(
    waveform: np.ndarray,
    target_length: int,
    mode: str = "random",  # "left", "right", or "random"
) -> np.ndarray:
    """Pad with silence to *target_length* samples."""
    current = waveform.shape[-1]
    if current >= target_length:
        return waveform
    pad_total = target_length - current
    if mode == "left":
        left, right = pad_total, 0
    elif mode == "right":
        left, right = 0, pad_total
    else:  # random
        left = random.randint(0, pad_total)
        right = pad_total - left
    pad_width = ((0, 0), (left, right)) if waveform.ndim == 2 else ((left, right),)
    return np.pad(waveform, pad_width)


def time_mask(
    waveform: np.ndarray,
    sample_rate: int,
    max_mask_sec: float = 0.05,
    num_masks: int = 1,
) -> np.ndarray:
    """Zero‑out *num_masks* random segments (<= max_mask_sec each)."""
    mask_len = int(max_mask_sec * sample_rate)
    out = waveform.copy()
    for _ in range(num_masks):
        if out.shape[-1] <= mask_len:
            continue
        start = random.randint(0, out.shape[-1] - mask_len)
        end = start + mask_len
        if out.ndim == 1:
            out[start:end] = 0
        else:
            out[:, start:end] = 0
    return out


# -----------------------------------------------------------------------------
# Augmentations operating on *spectrograms*
# -----------------------------------------------------------------------------

def frequency_mask(
    spec: np.ndarray,
    F: int = 15,
    num_masks: int = 1,
    replace_with_zero: bool = True,
) -> np.ndarray:
    """Randomly mask *F* frequency bins, SpecAugment style.

    Parameters
    ----------
    spec : np.ndarray
        Spectrogram (freq_bins, time_frames)
    F : int
        Maximum width of each mask (in bins)
    num_masks : int
        How many independent masks to apply.
    replace_with_zero : bool
        If False, masked bins are set to the mean instead of 0.
    """
    cloned = spec.copy()
    num_freqs = cloned.shape[0]
    for _ in range(num_masks):
        f = random.randint(0, F)
        f0 = random.randint(0, max(0, num_freqs - f))
        if replace_with_zero:
            cloned[f0 : f0 + f, :] = 0
        else:
            cloned[f0 : f0 + f, :] = cloned.mean()
    return cloned
