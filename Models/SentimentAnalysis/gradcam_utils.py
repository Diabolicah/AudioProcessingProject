"""Grad-CAM sweep over RAVDESS (book, sections 4.3 and 5.1.4).

Produces, per emotion and actor, a 3-row figure: the Grad-CAM heat map
(Figure 4), the spectrogram masked to the most important regions, and the
cross-correlation map for the emotions that have a hand-built kernel
(Figures 13-14).

    python gradcam_utils.py --model <model.pt> --attributes <prob_vector.csv>
    python gradcam_utils.py --model <model.pt>          # sweep every valid wav

Everything used to run at module import, with `Models.SentimentAnalysis.*`
imports that do not resolve when the package directory is the working
directory. It is now a function behind `if __name__ == "__main__"`.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.signal import correlate2d

from ConstPaths import RavdessPaths
from Preprocess import audio_to_mel_spectrogram
from PreprocessParams import HOP_LENGTH, MAX_SPECTOGRAM_DURATION_IN_SECONDS, SAMPLE_RATE
from audio_dataset import EmotionSpecDataset
from correlation_kernel_playground import kernel_list
from gradcam_initilaization import (GRADCAM_MASK_QUANTILE, GRADCAM_MIN_CONFIDENCE,
                                    get_device, index_emotion_mapping, is_valid_ravdess_file,
                                    label_emotion_mapping, load_model)

DEFAULT_OUTPUT_DIR = Path("Benchmark_Results") / "Summary_By_Actor_New"


def recordings_from_attributes(attributes_csv: Path,
                               min_confidence: float = GRADCAM_MIN_CONFIDENCE) -> list[Path]:
    """Confidently-classified recordings, read from a prob_vector.py CSV.

    Section 5.1.4 restricts Grad-CAM to samples the model classified with ~99%
    SoftMax confidence; that filter is applied here rather than being implied by
    a pre-filtered CSV.
    """
    df = pd.read_csv(attributes_csv, quoting=csv.QUOTE_NONE, encoding='utf-8',
                     engine='python', dtype=str)
    if "predicted_probability" in df.columns:
        confidence = pd.to_numeric(df["predicted_probability"], errors="coerce")
        df = df[confidence >= min_confidence]

    recordings = []
    for wav in df['path']:
        wav_path = Path(str(wav).replace("\\", "/").strip())
        if is_valid_ravdess_file(wav_path):
            recordings.append(wav_path)
    return recordings


def recordings_from_disk(root: Path = RavdessPaths.AUDIO_ORIGINAL_DATA) -> list[Path]:
    return sorted(wav for wav in Path(root).rglob("*.wav") if is_valid_ravdess_file(wav))


def group_by_emotion_actor(recordings: Sequence[Path],
                           actors_to_include: Optional[Sequence[str]] = None) -> dict:
    """emotion -> actor -> (statement, repetition) -> wav path."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for wav_path in recordings:
        parts = wav_path.stem.split("-")
        if len(parts) != 7:
            continue
        emotion_idx, statement, repetition, actor = parts[2], parts[4], parts[5], parts[6]
        emotion = index_emotion_mapping[emotion_idx]
        if actors_to_include is not None and actor not in actors_to_include:
            continue
        grouped[emotion][actor][(statement, repetition)] = wav_path
    return grouped


def run_gradcam_sweep(model, recordings: Sequence[Path], output_dir: Path = DEFAULT_OUTPUT_DIR,
                      mask_quantile: float = GRADCAM_MASK_QUANTILE, device=None) -> list[Path]:
    """Render one composite figure per (emotion, actor). Returns the saved paths."""
    device = device or get_device()
    emotion_to_actor_sentence_repetition = group_by_emotion_actor(recordings)

    # Build the CAM object once instead of once per sample: it registers forward
    # and backward hooks on the target layer, and re-creating it in the inner
    # loop stacked a new pair of hooks on every iteration.
    target_layers = [model.module3.blocks[-1].conv2]
    cam = GradCAM(model=model, target_layers=target_layers)

    saved_paths: list[Path] = []

    for emotion, actor_dict in emotion_to_actor_sentence_repetition.items():
        print(f"\n===> Processing emotion: {emotion}")
        plot_cc = emotion in kernel_list   # whether to plot the correlation map

        for actor, combo_dict in actor_dict.items():
            print(f"  Actor {actor}")
            n_cols = max(len(combo_dict), 1)
            fig, axes = plt.subplots(3, n_cols, figsize=(7.5 * n_cols, 12), sharex=True, squeeze=False)
            fig.suptitle(f"{emotion.capitalize()} – Actor {actor}", fontsize=18, y=0.98)

            sorted_keys = sorted(combo_dict.keys(), key=lambda x: (x[0], x[1]))  # (statement, repetition)

            img2 = None
            img3 = None
            for col_idx, (statement, repetition) in enumerate(sorted_keys):
                wav_path = combo_dict[(statement, repetition)]
                print(f"    Statement {statement}, Repetition {repetition} -> {wav_path.name}")
                spec_tensor, _ = EmotionSpecDataset([(wav_path, emotion)])[0]
                input_tensor = spec_tensor.unsqueeze(0).to(device)

                pred_idx = model(input_tensor).argmax(dim=1).item()
                cam_mask = cam(input_tensor=input_tensor,
                               targets=[ClassifierOutputTarget(pred_idx)],
                               aug_smooth=True,
                               eigen_smooth=True)[0]

                raw_spec = audio_to_mel_spectrogram(
                    file_path=wav_path,
                    max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS).astype("float32")
                # np.ptp(...) rather than raw_spec.ptp(): the ndarray method was
                # removed in NumPy 2.0.
                raw_norm = (raw_spec - raw_spec.min()) / (np.ptp(raw_spec) + 1e-6)
                rgb_base = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)
                overlay = show_cam_on_image(rgb_base, cam_mask, use_rgb=True, image_weight=0)

                axes[0, col_idx].imshow(
                    overlay, origin="lower", aspect="auto",
                    extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2],
                )
                axes[0][col_idx].set_title(
                    f"Statement: {statement} Repetition: {repetition} "
                    f"→ {label_emotion_mapping.get(pred_idx, pred_idx)}"
                )
                axes[0][col_idx].set_xlabel("Time (s)")
                axes[0][col_idx].set_ylabel("Freq (Hz)")

                # Spectrogram masked down to the most important Grad-CAM regions
                alpha_mask = np.clip(cam_mask, 0, 1)
                threshold = np.quantile(alpha_mask, mask_quantile)
                strong_activation_mask = (alpha_mask >= threshold).astype(np.float32)

                # Apply the mask to the original spectrogram (not RGB)
                masked_spec = np.where(strong_activation_mask > 0, raw_spec, -80.0)

                img2 = librosa.display.specshow(
                    masked_spec,
                    sr=SAMPLE_RATE,
                    hop_length=HOP_LENGTH,
                    x_axis='time',
                    y_axis='linear',
                    ax=axes[1, col_idx],
                    cmap='magma',
                )

                axes[1, col_idx].set_title(f"Masked at the top {(1 - mask_quantile) * 100:.0f}% of activations")
                axes[1, col_idx].set_xlabel("Time (s)")
                axes[1, col_idx].set_ylabel("Freq (Hz)")

                if plot_cc:
                    masked_spec_shift = masked_spec - np.min(masked_spec)
                    corr = correlate2d(masked_spec_shift, kernel_list[emotion])

                    # Normalize correlation to [-1,1] for visualization
                    corr /= np.max(np.abs(corr)) + 1e-12

                    img3 = axes[2, col_idx].imshow(
                        corr, aspect='auto', origin='lower', cmap='RdBu_r',
                        extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2])
                    axes[2, col_idx].set_title("Cross-Correlation with emotion kernel")

            if plot_cc and img3 is not None:
                plt.colorbar(img3, ax=axes[2, :], orientation='horizontal', label='Correlation')
            if img2 is not None:
                cbar_spec = fig.colorbar(img2, ax=axes[1, :], orientation='horizontal')
                cbar_spec.set_label("Spectrogram Magnitude (dB)")

            save_folder = Path(output_dir) / actor
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / f"Emotion_{emotion}_overview.png"
            plt.savefig(save_path)
            plt.close(fig)
            saved_paths.append(save_path)
            print(f"Saved: {save_path}")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="trained RAVDESS .pt classifier")
    parser.add_argument("--attributes", type=Path, default=None,
                        help="prob_vector.py CSV; restricts the sweep to confident predictions")
    parser.add_argument("--min-confidence", type=float, default=GRADCAM_MIN_CONFIDENCE)
    parser.add_argument("--mask-quantile", type=float, default=GRADCAM_MASK_QUANTILE,
                        help="keep activations above this quantile (default: the book's 0.85)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.attributes is not None:
        wavs = recordings_from_attributes(args.attributes, args.min_confidence)
    else:
        wavs = recordings_from_disk()
    print(f"{len(wavs)} recordings selected")

    run_gradcam_sweep(load_model(args.model), wavs,
                      output_dir=args.out, mask_quantile=args.mask_quantile)
