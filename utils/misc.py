# -*- coding: utf-8 -*-
"""
Miscellaneous utility functions for V3Cell.

Contents:
  - load_config:      Load a YAML configuration file.
  - set_seed:         Set random seeds for reproducibility.
  - save_video_as_gif: Render a video tensor to an animated GIF.
  - save_video_as_mp4: Render a video tensor to MP4 (requires imageio).
  - count_parameters: Count trainable parameters in a PyTorch model.
  - format_time:      Human-readable elapsed time string.
  - EMAModel:         Exponential moving average of model weights.
"""

import os
import random
from pathlib import Path
from typing import Union, Optional

import yaml
import numpy as np
import torch
from PIL import Image


def load_config(config_path: Union[str, Path]) -> dict:
    """
    Load a YAML configuration file into a Python dictionary.

    Args:
        config_path: Path to the .yaml configuration file.

    Returns:
        config: Nested dictionary of configuration values.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """
    Set random seeds for Python, NumPy and PyTorch to ensure reproducibility.

    Also disables cuDNN's non-deterministic algorithms (slightly slower but
    fully reproducible results).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_video_as_gif(
    video: torch.Tensor,
    path: Union[str, Path],
    fps: int = 10,
    loop: int = 0,
):
    """
    Save a video tensor as an animated GIF.

    Handles both [T, C, H, W] and [T, H, W, C] layouts and both [-1, 1]
    and [0, 1] value ranges.

    Args:
        video: Video frames — tensor or numpy array.
        path:  Output file path (should end in .gif).
        fps:   Frames per second of the output GIF.
        loop:  Number of times the GIF loops (0 = infinite).
    """
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()

    # Convert [T, C, H, W] → [T, H, W, C]
    if video.ndim == 4 and video.shape[1] in [1, 3]:
        video = np.transpose(video, (0, 2, 3, 1))

    # Normalise to [0, 255]
    if video.min() < 0:  # [-1, 1] range
        video = (video + 1) / 2
    video = (video * 255).clip(0, 255).astype(np.uint8)

    # Ensure 3-channel RGB
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)

    frames = [Image.fromarray(frame) for frame in video]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=loop,
    )


def save_video_as_mp4(
    video: torch.Tensor,
    path: Union[str, Path],
    fps: int = 10,
):
    """
    Save a video tensor as an MP4 file (requires imageio with ffmpeg).

    Falls back to GIF if imageio is not available.

    Args:
        video: [T, C, H, W] video tensor.
        path:  Output file path (should end in .mp4).
        fps:   Frames per second.
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed — saving as GIF instead.")
        save_video_as_gif(video, str(path).replace('.mp4', '.gif'), fps)
        return

    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()

    if video.ndim == 4 and video.shape[1] in [1, 3]:
        video = np.transpose(video, (0, 2, 3, 1))

    if video.min() < 0:
        video = (video + 1) / 2
    video = (video * 255).clip(0, 255).astype(np.uint8)

    imageio.mimwrite(path, video, fps=fps)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: Any torch.nn.Module.

    Returns:
        Total number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format an elapsed time in seconds as a human-readable string.

    Examples:
        42.5  → '42.5s'
        125   → '2m 5s'
        3700  → '1h 1m'

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class EMAModel:
    """
    Exponential Moving Average (EMA) of model weights.

    Maintains a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Using EMA weights at inference time typically produces smoother, higher-
    quality generated outputs compared to the raw training weights.

    Usage:
        ema = EMAModel(generator, decay=0.999)
        for step in training_loop:
            update weights ...
            ema.update(generator)
        ema.apply(generator)   # swap in EMA weights for evaluation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.device = device

        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)

    def update(self, model: torch.nn.Module):
        """Update shadow weights from the current model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module):
        """Replace model parameters with the EMA shadow weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
