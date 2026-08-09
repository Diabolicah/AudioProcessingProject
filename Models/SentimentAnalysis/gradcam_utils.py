import csv
from collections import defaultdict

import librosa
import pandas as pd
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.signal import correlate2d

from Models.SentimentAnalysis.ConstPaths import RavdessPaths
from Models.SentimentAnalysis.Preprocess import audio_to_mel_spectrogram
from Models.SentimentAnalysis.audio_dataset import EmotionSpecDataset
from Models.SentimentAnalysis.correlation_kernel_playground import kernel_list
from Models.SentimentAnalysis.gradcam_initilaization import *

PANDAS_FLAG: bool | None = True
RECORDINGS_TO_PROCESS = []
if PANDAS_FLAG is True:
    df = pd.read_csv(Path("new_attributes/99_attributes_new_split.csv"), usecols=['path'], quoting=csv.QUOTE_NONE,
                     encoding='utf-8', engine='python', dtype=str)
    for wav in df['path']:
        wav_path = Path(wav.replace("\\", "/").strip())
        if is_valid_ravdess_file(wav_path):
            RECORDINGS_TO_PROCESS.append(wav_path)
else:
    RECORDINGS_TO_PROCESS = [
        wav for wav in RavdessPaths.AUDIO_ORIGINAL_DATA.rglob("*.wav")
        if is_valid_ravdess_file(wav)
    ]

# --------------------------------------------------------------
# 3.  Group by emotion, actor, statement+repetition => returns "emotion_to_actor_sentence_repetition[emotion][actor][key] = wav_path"
# --------------------------------------------------------------
emotion_to_actor_sentence_repetition = defaultdict(lambda: defaultdict(dict))

for wav_path in RECORDINGS_TO_PROCESS:
    parts = wav_path.stem.split("-")
    if len(parts) != 7:
        continue
    emotion_idx, statement, repetition, actor = parts[2], parts[4], parts[5], parts[6]
    emotion = index_emotion_mapping[emotion_idx]
    key = (statement, repetition)

    if actor in ACTORS_TO_INCLUDE and emotion in index_emotion_mapping.values():
        emotion_to_actor_sentence_repetition[emotion][actor][key] = wav_path



# --------------------------------------------------------------
# 4.  Create and save composite plots per emotion-actor # goes through emotion_to_actor_sentence_repetition => plots ALL the composite graphs
# --------------------------------------------------------------

plot_cc = False
for emotion, actor_dict in emotion_to_actor_sentence_repetition.items():
    print(f"\n===> Processing emotion: {emotion}")

    plot_cc = True if emotion in kernel_list else False # whether to plot the correlation map or not

    for actor, combo_dict in actor_dict.items():
        print(f"  Actor {actor}")
        fig, axes = plt.subplots(3, 4, figsize=(30, 12), sharex=True)
        fig.suptitle(f"{emotion.capitalize()} – Actor {actor}", fontsize=18, y=0.98)

        sorted_keys = sorted(combo_dict.keys(), key=lambda x: (x[0], x[1]))  # (statement, repetition)

        img2 = None
        img3 = None
        for col_idx, (statement, repetition) in enumerate(sorted_keys):
            wav_path = combo_dict[(statement, repetition)]
            print(f"    Statement {statement}, Repetition {repetition} → {wav_path.name}")
            spec_tensor, _ = EmotionSpecDataset({(wav_path, emotion)})[0]
            input_tensor = spec_tensor.unsqueeze(0).to(device)

            target_layers = [model.module3.blocks[-1].conv2]
            cam = GradCAM(model=model, target_layers=target_layers)
            pred_idx = model(input_tensor).argmax(dim=1).item()
            cam_mask = cam(input_tensor=input_tensor,
                           targets=[ClassifierOutputTarget(pred_idx)],
                           aug_smooth=True,
                           eigen_smooth=True)[0]

            # high_activation_regions = extract_important_time_regions(
            #     cam_mask, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, threshold_quantile=0.75)

            # y, sr = librosa.load(wav_path, mono=True, sr=SAMPLE_RATE)
            # if len(y) < MAX_SAMPLES:
            #     y = np.pad(y, (0, MAX_SAMPLES - len(y)))
            # else:
            #     y = y[:MAX_SAMPLES]
            # rms = librosa.feature.rms(y=y, frame_length=WINDOW_LENGTH, hop_length=HOP_LENGTH)[0]
            # rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP_LENGTH)
            #
            # f0, _, _ = librosa.pyin(
            #     y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            #     sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
            # f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)

            raw_spec = audio_to_mel_spectrogram(
                file_path=wav_path, max_length_in_seconds=MAX_SPECTOGRAM_DURATION_IN_SECONDS,
                normalization_fn=lambda x: x).astype("float32")
            raw_norm = (raw_spec - raw_spec.min()) / (raw_spec.ptp() + 1e-6)
            rgb_base = np.stack([raw_norm] * 3, axis=-1).astype(np.float32)
            overlay = show_cam_on_image(rgb_base, cam_mask, use_rgb=True, image_weight=0)

            # axes[0][col_idx].set_title(f"Stmt {statement} Rep {repetition}")
            # plot_segmented_line(axes[0][col_idx], rms_times, rms, high_activation_regions,
            #                     base_color='lightgray', highlight_color='crimson', linewidth=2)
            # axes[0][col_idx].set_ylabel("Volume")
            # axes[0][col_idx].set_ylim(0, 0.5)
            # axes[0][col_idx].grid(True)
            #
            # plot_segmented_line(axes[1][col_idx], f0_times, f0, high_activation_regions,
            #                     base_color='lightgray', highlight_color='black', linewidth=2)
            # axes[1][col_idx].set_ylabel("F0 (Hz)")
            # axes[1][col_idx].set_ylim(0, 500)
            # axes[1][col_idx].grid(True)

            axes[0, col_idx].imshow(
                overlay, origin="lower", aspect="auto",
                extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2],
            )
            emotion_classified = label_emotion_mapping[pred_idx]
            axes[0][col_idx].set_title(f"Statement: {statement} Repetition: {repetition}")
            axes[0][col_idx].set_xlabel("Time (s)")
            axes[0][col_idx].set_ylabel("Freq (Hz)")

            # NEW: Plot CAM-filtered spectrogram
            alpha_mask = np.clip(cam_mask, 0, 1)

            # Keep only yellow-red regions (~ top 30% of activation)
            threshold = np.quantile(alpha_mask, 0.92) #84%, 86%, 88%, 90%(?),
            strong_activation_mask = (alpha_mask >= threshold).astype(np.float32)

            # Apply the mask to the original spectrogram (not RGB)
            masked_spec = np.where(strong_activation_mask > 0, raw_spec, -80.0)

            # Plot using librosa with a colormap (e.g., magma, viridis)
            img2 = librosa.display.specshow(
                masked_spec,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                x_axis='time',
                y_axis='linear',
                ax=axes[1, col_idx],
                cmap='magma',  # or 'viridis', 'inferno', etc.
            )

            axes[1, col_idx].set_title("Filtered by Attention (Top 30%)")
            axes[1, col_idx].set_xlabel("Time (s)")
            axes[1, col_idx].set_ylabel("Freq (Hz)")

            if plot_cc:
                masked_spec_shift = masked_spec - np.min(masked_spec)
                corr = correlate2d(masked_spec_shift, kernel_list[emotion])

                # Normalize correlation to [-1,1] for visualization
                corr /= np.max(np.abs(corr)) + 1e-12

                # --- 5. Plot cross-correlation map ---
                img3 = axes[2, col_idx].imshow(corr, aspect='auto', origin='lower', cmap='RdBu_r', extent=[0, raw_spec.shape[1] * HOP_LENGTH / SAMPLE_RATE, 0, SAMPLE_RATE // 2])
                axes[2, col_idx].set_title("Cross-Correlation with Smooth Flat Kernel")
                #if col_idx == axes.shape[1] - 1:  # last column

        if plot_cc:
            cbar_corr = plt.colorbar(img3, ax=axes[2, :], orientation='horizontal', label='Correlation')
        cbar_spec = fig.colorbar(img2, ax=axes[1, :], orientation='horizontal')
        cbar_spec.set_label("Spectrogram Magnitude (dB)")
        save_folder = Path("Benchmark_Results") / "Summary_By_Actor_New" / actor
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / f"Emotion_{emotion}_overview.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")

# --------------------------------------------------------------
# 5. PCA Summary for All Recordings – RMS and F0 by Emotion
# --------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused but required for 3D plot
#
# rms_vectors = []
# f0_vectors = []
# emotion_labels = []
