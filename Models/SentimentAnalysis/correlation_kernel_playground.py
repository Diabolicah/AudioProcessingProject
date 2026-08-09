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
    'disgust' : np.array([
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
#correlation_kernel /= correlation_kernel.sum()


# # Kernel size
# freq_bins = 40   # vertical axis (frequency)
# time_bins = 20   # horizontal axis (time)
#
# # Create smooth frequency profile: almost flat, small gentle slope
# freq_profile = np.linspace(1, 0.9, freq_bins)[:, np.newaxis]  # very gentle high→low
#
# # Smooth time modulation: soft sine wave
# time_profile = np.sin(np.linspace(0, np.pi, time_bins))[np.newaxis, :]
#
# # Combine profiles to get 2D kernel
# kernel = freq_profile * time_profile  # element-wise multiplication
#
# # Normalize: zero-mean and unit-norm
# kernel -= kernel.mean()
# kernel /= np.linalg.norm(kernel) + 1e-12
#
# # Visualize
plt.figure(figsize=(6,4))
plt.imshow(kernel_list['surprised'], aspect='auto', origin='lower', cmap='RdBu_r')
plt.colorbar(label='Amplitude')
plt.title("Smooth Flat-Frequency Kernel (40x20)")
plt.xlabel("Time bins")
plt.ylabel("Frequency bins")
plt.show()