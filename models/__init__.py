# -*- coding: utf-8 -*-
"""
V3Cell Models (MoCoGAN-HD specification)

Components:
- MotionRNN: Temporal latent sequence generator (R_psi in the paper)
- V3CellGenerator: Main video generator (wraps the frozen temporal organoid
  generator + MotionRNN + Encoder) used to construct a temporal virtual
  organoid from early observed frames
- ImageDiscriminator: 2D frame-pair discriminator
- VideoDiscriminator: 3D multi-scale video discriminator
- ImageEncoder, SequenceEncoder: Image/sequence encoders for conditional generation
- FateClassifier: Fate classifier Phi used for early-frame developmental
  fate prediction
"""

from .temporal_organoid_generator import Generator as TemporalOrganoidGenerator, Discriminator as TemporalOrganoidDiscriminator
from .motion_rnn import MotionRNN, MotionRNNWithPCA
from .video_generator import V3CellGenerator
from .discriminators import (
    ImageDiscriminator,
    VideoDiscriminator,
    prepare_image_disc_input,
    prepare_video_disc_input,
)
from .encoder import ImageEncoder, SequenceEncoder
from .fate_classifier import FateClassifier

# Backward-compatibility alias so that older scripts using VideoGenerator still work.
VideoGenerator = V3CellGenerator

__all__ = [
    # R3GAN network architecture, frozen and reused inside Stage 2 to decode
    # each latent into a frame of the temporal virtual organoid (also reused,
    # via r3gan/training/networks.py, to train the per-lineage Stage 1 static
    # generators from scratch — this module is lineage-agnostic)
    'TemporalOrganoidGenerator',
    'TemporalOrganoidDiscriminator',
    # V3Cell video generation (Stage 2 — dynamic modeling)
    'MotionRNN',
    'MotionRNNWithPCA',
    'V3CellGenerator',
    'VideoGenerator',        # backward-compatibility alias
    'ImageDiscriminator',
    'VideoDiscriminator',
    'prepare_image_disc_input',
    'prepare_video_disc_input',
    'ImageEncoder',
    'SequenceEncoder',
    'FateClassifier',
]
