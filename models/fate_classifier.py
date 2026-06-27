# -*- coding: utf-8 -*-
"""
FateClassifier: predict organoid developmental fate from early observed frames.

This is the 'Fate Predict' block shown in Figure c of the V3Cell paper.

Usage pattern
-------------
1. Train a classifier separately (e.g. using experiment2.4/exp_fate_confidence.py
   which trains a ResNet50 on organoid frames).

2. Save the classifier checkpoint (state_dict of the ResNet50 model).

3. Load it here as a frozen FateClassifier and attach to V3CellGenerator:

       predictor = FateClassifier.from_resnet50_checkpoint('path/to/classifier.pt')
       generator = V3CellGenerator(..., use_fate_classifier=True)
       generator.fate_classifier = predictor

4. During inference, V3CellGenerator.predict_from_frames() will automatically
   call FateClassifier on the early frames and use the predicted class vector C
   instead of a manually supplied ground-truth label.

Architecture
------------
    early_frames  [B, T_in, 3, H, W]
        → per-frame ResNet50 (frozen, pre-trained)
        → softmax probabilities per frame  [B, T_in, c_dim]
        → mean over T_in                   [B, c_dim]   ← class condition C

The mean pooling aggregation is simple but effective: it averages the per-frame
fate signal over the observed prefix, giving a robust estimate even with noisy
early frames.

Alternatively, the predictor can return the probabilities for the last frame only
(set aggregation='last'), or the max-confidence frame (aggregation='max').
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FateClassifier(nn.Module):
    """
    Frozen fate classifier wrapper.

    Wraps a pre-trained image classifier (default: ResNet50 fine-tuned on
    organoid frames) so it can be used as the 'Fate Predict' block inside
    V3CellGenerator.

    The classifier is always frozen — it is never updated during V3Cell training.

    Args:
        backbone:     A torch.nn.Module that maps [B, 3, H, W] → [B, c_dim] logits.
        c_dim:        Number of fate classes.
        img_size:     Size the frames are resized to before feeding to backbone (224
                      for standard ResNet/ViT backbones).
        aggregation:  How to combine per-frame predictions over T_in frames.
                      'mean'  — average probabilities (default, most robust)
                      'last'  — use only the last early frame
                      'max'   — use the frame with highest max-class confidence
        temperature:  Softmax temperature applied to backbone logits (default 1.0).
    """

    def __init__(
        self,
        backbone: nn.Module,
        c_dim: int = 3,
        img_size: int = 224,
        aggregation: str = 'mean',
        temperature: float = 1.0,
    ):
        super().__init__()

        assert aggregation in ('mean', 'last', 'max'), \
            f"aggregation must be 'mean', 'last', or 'max', got '{aggregation}'"

        self.backbone = backbone
        self.c_dim = c_dim
        self.img_size = img_size
        self.aggregation = aggregation
        self.register_buffer('temperature', torch.tensor(temperature))

        # Freeze all backbone parameters — FateClassifier is never trained
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Override train() to keep backbone permanently in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_resnet50_checkpoint(
        cls,
        checkpoint_path: str,
        c_dim: int = 3,
        img_size: int = 224,
        aggregation: str = 'mean',
        temperature: float = 1.0,
    ) -> 'FateClassifier':
        """
        Build a FateClassifier from a ResNet50 classifier checkpoint.

        Expects the checkpoint to be a dict with key 'model_state_dict' or a
        plain state_dict of the ResNet50 (as saved by exp_fate_confidence.py
        using torch.save({'model': model.state_dict(), ...}) or similar).

        Args:
            checkpoint_path: Path to the saved ResNet50 checkpoint (.pt / .pth).
            c_dim:           Number of output classes.
            img_size:        Resize spatial size for input frames (default 224).
            aggregation:     Per-frame aggregation strategy.
            temperature:     Softmax temperature.

        Returns:
            FateClassifier instance (backbone frozen).
        """
        import torchvision.models as tv_models

        backbone = tv_models.resnet50(weights=None)
        backbone.fc = nn.Linear(2048, c_dim)

        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Support several common checkpoint formats
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                # Assume the dict itself is a state_dict
                state_dict = ckpt
        else:
            state_dict = ckpt

        backbone.load_state_dict(state_dict, strict=True)

        return cls(
            backbone=backbone,
            c_dim=c_dim,
            img_size=img_size,
            aggregation=aggregation,
            temperature=temperature,
        )

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        c_dim: int = 3,
        img_size: int = 224,
        aggregation: str = 'mean',
        temperature: float = 1.0,
    ) -> 'FateClassifier':
        """
        Build a FateClassifier from any pre-trained backbone that outputs logits.

        Useful for custom backbones or backbones already loaded in memory.
        """
        return cls(
            backbone=backbone,
            c_dim=c_dim,
            img_size=img_size,
            aggregation=aggregation,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Normalise frames from [-1, 1] to ImageNet-normalised [0, 1] and resize.

        Args:
            frames: [N, 3, H, W] in [-1, 1].

        Returns:
            Preprocessed frames [N, 3, img_size, img_size].
        """
        # [-1, 1] → [0, 1]
        x = (frames + 1.0) / 2.0

        # Resize to backbone's expected input size
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False,
            )

        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def forward(self, early_frames: torch.Tensor) -> torch.Tensor:
        """
        Predict fate class probabilities from early observed frames.

        Args:
            early_frames: [B, T_in, 3, H, W] — early development frames in [-1, 1].

        Returns:
            probs: [B, c_dim] — class probability vector (soft one-hot condition C).
        """
        B, T_in, C, H, W = early_frames.shape

        # Encode all frames together in one batch pass
        frames_flat = early_frames.reshape(B * T_in, C, H, W)
        frames_proc = self._preprocess(frames_flat)          # [B*T, 3, img_size, img_size]

        logits_flat = self.backbone(frames_proc)             # [B*T, c_dim]
        probs_flat  = F.softmax(logits_flat / self.temperature, dim=-1)  # [B*T, c_dim]

        probs = probs_flat.reshape(B, T_in, self.c_dim)      # [B, T_in, c_dim]

        # Aggregate over the T_in early frames
        if self.aggregation == 'mean':
            out = probs.mean(dim=1)                          # [B, c_dim]
        elif self.aggregation == 'last':
            out = probs[:, -1, :]                            # [B, c_dim]
        elif self.aggregation == 'max':
            # Frame with the highest confidence (max probability mass on winner class)
            confidence = probs.max(dim=-1).values            # [B, T_in]
            best_t = confidence.argmax(dim=1)                # [B]
            out = probs[torch.arange(B, device=probs.device), best_t]  # [B, c_dim]

        return out

    def predict_hard(self, early_frames: torch.Tensor) -> torch.Tensor:
        """
        Return a one-hot hard prediction (argmax of probabilities).

        Args:
            early_frames: [B, T_in, 3, H, W].

        Returns:
            one_hot: [B, c_dim].
        """
        probs = self.forward(early_frames)
        indices = probs.argmax(dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
        return one_hot

    def set_temperature(self, temperature: float):
        """Adjust softmax temperature at runtime."""
        self.temperature.fill_(temperature)
