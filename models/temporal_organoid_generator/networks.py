"""
R3GAN wrapper networks used to instantiate the frozen image generator that
V3CellGenerator decodes through when constructing a temporal virtual organoid
(Stage 2), loaded from a Stage 1 R3GAN checkpoint.

These thin wrapper classes bridge the checkpoint serialisation format produced
by the original r3gan/train.py (which uses key names like 'NoiseDimension',
'FP16Stages', 'ConditionDimension') and the core StyleGAN3-based implementation
in models/temporal_organoid_generator/core.py.

The wrappers expose the standard forward(z, c) signature used throughout V3Cell
and store .z_dim, .c_dim and .img_resolution attributes so that V3CellGenerator
can inspect the generator without unwrapping internals.

Note: this same module is also re-exported (via the backward-compatibility
shim at r3gan/training/networks.py) as the network architecture used to train
the per-lineage Stage 1 static generators (colon/stomach/lung) from scratch.
This module is lineage-agnostic; in the paper's reported amniotic-sac
experiments, an instance of it frozen for Stage 2 is referred to as G_AS.
"""

import torch
import torch.nn as nn
import copy

from . import core


class Generator(nn.Module):
    """
    R3GAN generator wrapper.

    Translates the checkpoint keyword arguments produced by r3gan/train.py
    into the format expected by core.Generator, and handles the bfloat16
    FP16 stage configuration used by some R3GAN presets.
    """

    def __init__(self, *args, **kw):
        super().__init__()

        # Strip wrapper-specific keys before passing the rest to the core.
        config = copy.deepcopy(kw)
        fp16_stages = config.pop('FP16Stages')
        c_dim = config.pop('c_dim')
        img_resolution = config.pop('img_resolution')

        # The core Generator uses 'ConditionDimension' only when c_dim > 0.
        if c_dim != 0:
            config['ConditionDimension'] = c_dim

        self.Model = core.Generator(*args, **config)
        self.z_dim = kw['NoiseDimension']
        self.c_dim = c_dim
        self.img_resolution = img_resolution

        # Enable bfloat16 for the requested stages (memory / speed trade-off).
        for stage_idx in fp16_stages:
            self.Model.MainLayers[stage_idx].DataType = torch.bfloat16

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise and condition.

        Args:
            z: [B, z_dim] — noise vectors sampled from N(0, I).
            c: [B, c_dim] — class-condition vectors (one-hot or soft).

        Returns:
            images: [B, C, H, W] — generated images (may be bfloat16 in FP16 stages).
        """
        return self.Model(z, c)


class Discriminator(nn.Module):
    """
    R3GAN discriminator wrapper.

    Mirrors the Generator wrapper: translates checkpoint keyword arguments and
    configures bfloat16 stages.  Used only during Stage 1 (static image GAN)
    training; not used in the V3Cell Stage 2 video training.
    """

    def __init__(self, *args, **kw):
        super().__init__()

        config = copy.deepcopy(kw)
        fp16_stages = config.pop('FP16Stages')
        c_dim = config.pop('c_dim')
        config.pop('img_resolution')

        if c_dim != 0:
            config['ConditionDimension'] = c_dim

        self.Model = core.Discriminator(*args, **config)

        for stage_idx in fp16_stages:
            self.Model.MainLayers[stage_idx].DataType = torch.bfloat16

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute real/fake logits.

        Args:
            x: [B, C, H, W] — real or generated images.
            c: [B, c_dim]   — class-condition vectors.

        Returns:
            logits: [B, 1] — discriminator scores.
        """
        return self.Model(x, c)
