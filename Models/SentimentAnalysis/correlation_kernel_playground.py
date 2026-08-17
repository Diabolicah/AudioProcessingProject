import numpy as np
import matplotlib.pyplot as plt


def wave_kernel(freq_bins=20, time_bins=40) -> np.ndarray:
    # Frequency profile: flat (all ones)
    freq_profile = np.ones((freq_bins, 1), dtype=np.float32)

    # Time profile: simple sine wave going high→low→high
    time_profile = np.sin(np.linspace(0, np.pi, time_bins))[np.newaxis, :]  # single wave

    # Combine
    kernel = freq_profile * time_profile  # broadcast over freq

    # Optional: zero-mean & normalize
    kernel -= kernel.mean()
    kernel /= np.linalg.norm(kernel) + 1e-12

    return kernel


kernel_list = dict({
    'calm': np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float),
    'surprised': wave_kernel(),
    'happy' : np.array([
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [ 92,  92,  92,  92,  92,  92,   0,   0,   0,   0],
        [100, 100, 100, 100, 100, 100,  83,   0,   0,   0],
        [ 95,  95,  95,  95,  95,  95,  88,  81,   0,   0],
        [  0,   0,   0,   0,   0,   0,  83,  86,  79,   0],
        [  0,   0,   0,   0,   0,   0,   0,  81,  84,  77],
        [  0,   0,   0,   0,   0,   0,   0,   0,  79,  82],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,  77]
        ]),
    'angry' : np.array([
        [ 0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0, 90,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, 85, 90, 85,  0,  0,  0,  0],
        [ 0,  0,  0,  0, 90,  90, 90,  0,  0,  0,  0],
        [ 0,  0,  0, 80, 90,  90, 90, 80,  0,  0,  0],
        [ 0,  0,  0, 80, 90,  80, 90, 80,  0,  0,  0],
        [ 0,  0, 70, 75, 75,  50, 75, 75, 70,  0,  0],
        [ 0,  0, 70, 75, 60,  10, 60, 75, 70,  0,  0],
        [ 0,  60, 50, 36, 40,  0, 40, 36, 50,  60,  0],
        [ 0,  4, 14, 34, 30,  0, 30, 34, 14,  4,  0]
        ]),
    'sad' : np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=float),
    # Key must be the canonical label spelling ("disgusted", not "disgust"),
    # otherwise gradcam_utils never finds this kernel and silently skips the
    # cross-correlation row for disgust.
    'disgusted' : np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=float),
    'fearful' : np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=float)
})

def normalize_sum(temp_kernel):
    return temp_kernel / temp_kernel.sum()

kernel_list = {emotion: normalize_sum(kernel) for emotion, kernel in kernel_list.items()}
# Normalize so correlation scores are interpretable


def show_kernel(emotion: str = 'surprised') -> None:
    """Visualise one hand-built correlation kernel."""
    plt.figure(figsize=(6, 4))
    plt.imshow(kernel_list[emotion], aspect='auto', origin='lower', cmap='RdBu_r')
    plt.colorbar(label='Amplitude')
    plt.title(f"Correlation kernel: {emotion}")
    plt.xlabel("Time bins")
    plt.ylabel("Frequency bins")
    plt.show()


if __name__ == "__main__":
    # This used to run at import time, so `import correlation_kernel_playground`
    # (which gradcam_utils does) opened a blocking matplotlib window.
    show_kernel('surprised')