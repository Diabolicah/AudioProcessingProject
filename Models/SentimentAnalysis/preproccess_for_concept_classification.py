""" 
Preprocess a spectrogram for input to a vision backbone (e.g., ResNet), so it has the right size and normalization. 
Used to fine-tune vision backbones for our multi-label classification (given a mel spectrogram, predict what concepts are present).
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from Preprocess import audio_to_mel_spectrogram
from Visualizations import plot_mel_spectrogram

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

TensorLike = Union[torch.Tensor, Sequence[float]]

def prepare_spectrogram_for_backbone(
    spec: torch.Tensor,
    *,
    out_size: Tuple[int, int] = (224, 224),
    letterbox: bool = False,
    to_rgb: bool = True,
    mean: Optional[TensorLike] = IMAGENET_MEAN,   # e.g., [0.485, 0.456, 0.406]
    std: Optional[TensorLike]  = IMAGENET_STD,   # e.g., [0.229, 0.224, 0.225]
    normalize: bool = True,
    device: Optional[torch.device] = None,
    return_both: bool = True,
):
    """
    Resize a (H,W) or (C,H,W) spectrogram to `out_size`, optionally keep aspect ratio,
    optionally convert to RGB, and (optionally) normalize for an ML model with provided mean/std.

    Args:
        spec: (H,W), (1,H,W) or (3,H,W) float/uint tensor.
        out_size: (out_h, out_w) in pixels.
        letterbox: True → keep aspect ratio with padding; False → direct resize.
        to_rgb: True → ensure 3 channels by repeating if needed.
        mean, std: per-channel stats (len==1 or len==C). If None, skip normalization.
        normalize: apply (x - mean) / std if mean & std are provided.
        device: target device (defaults to spec.device if set, else CPU).
        return_both: True → returns (plot_img_[0..1], model_img_[normalized or 0..1])

    Returns:
        If return_both:
          plot_img:  (C,H,W) in [0,1] for visualization
          model_img: (C,H,W) normalized if mean/std provided & normalize=True, else [0,1]
        Else:
          model_img only.
    """
    if device is None:
        device = spec.device if hasattr(spec, "device") else torch.device("cpu")

    # Ensure (C,H,W) and float32
    x = spec
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1,H,W)
    assert x.ndim == 3 and x.shape[0] in (1, 3), "spec must be (H,W), (1,H,W), or (3,H,W)"
    x = x.to(torch.float32).to(device)

    # Min–max normalize to [0,1] for visualization & stable resizing
    vmin, vmax = x.min(), x.max()
    x01 = (x - vmin) / (vmax - vmin) if (vmax - vmin) > 0 else torch.zeros_like(x)

    out_h, out_w = out_size
    H, W = x01.shape[-2], x01.shape[-1]

    if letterbox:
        scale = min(out_h / H, out_w / W)
        newH, newW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
        x_res = F.interpolate(x01.unsqueeze(0), size=(newH, newW), mode="bilinear", align_corners=False).squeeze(0)
        padH, padW = out_h - newH, out_w - newW
        pad_top = padH // 2; pad_bottom = padH - pad_top
        pad_left = padW // 2; pad_right = padW - pad_left
        x_res = F.pad(x_res, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    else:
        x_res = F.interpolate(x01.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)

    # Channels
    if to_rgb and x_res.shape[0] == 1:
        plot_img = x_res.repeat(3, 1, 1)  # (3,H,W)
    else:
        plot_img = x_res

    # Prepare mean/std to shape (C,1,1) if provided
    def _prep_stats(arr: TensorLike, channels: int) -> torch.Tensor:
        t = torch.as_tensor(arr, dtype=torch.float32, device=device)
        if t.ndim == 1:
            if t.numel() == channels:
                t = t.view(channels, 1, 1)
            elif t.numel() == 1:
                t = t.view(1, 1, 1).repeat(channels, 1, 1)
            else:
                raise ValueError(f"Mean/Std length {t.numel()} does not match channels {channels} and is not 1.")
        elif t.ndim == 3:
            assert t.shape == (channels, 1, 1), f"Mean/Std must be (C,1,1) with C={channels}."
        else:
            raise ValueError("Mean/Std must be 1D (len==C or 1) or 3D (C,1,1).")
        return t

    channels = plot_img.shape[0]
    if normalize and (mean is not None) and (std is not None):
        mean_t = _prep_stats(mean, channels)
        std_t  = _prep_stats(std,  channels)
        model_img = (plot_img - mean_t) / std_t
    else:
        model_img = plot_img  # still in [0,1]

    return (plot_img, model_img) if return_both else model_img


if __name__ == "__main__":
    
    # ----------- Example usage ----------
    
    # load audio and pass it to audio_to_mel_spectrogram
    audio_path = Path(r'RAVDESS\original_data\Actor_02\03-01-02-01-01-01-02.wav')
    mel_spectrogram = audio_to_mel_spectrogram(audio_path)  
    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)

    plot_mel_spectrogram(mel_spectrogram, block=False)

    print(mel_spectrogram.shape)
    
    plot_img, model_img = prepare_spectrogram_for_backbone(mel_spectrogram, letterbox=False, return_both=True)

    #? plot with plot_mel_spectrogram (specshow)
    plot_mel_spectrogram(plot_img, block=True)
    # plt.show()

    # test that model_img = (plot_img - mean) / std
    reconstructed_model_img = (plot_img - IMAGENET_MEAN) / IMAGENET_STD
    print(torch.allclose(reconstructed_model_img, model_img, atol=1e-5))  # should be True

