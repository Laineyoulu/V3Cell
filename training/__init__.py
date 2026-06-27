# -*- coding: utf-8 -*-
"""
V3Cell Training Module (MoCoGAN-HD specification).

Training components:
  - V3CellTrainer:     Unconditional trainer — generates videos from noise z.
  - ConditionalTrainer: Conditional trainer — predicts future frames from early observations.
  - Loss functions:    Relativistic Average LSGAN, gradient penalties, MI loss.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from training.losses import (
    GANLoss,
    RelativisticAverageLSGAN,
    compute_gradient_penalty_3d,
    compute_gradient_penalty_2d,
    compute_mutual_information_loss,
)
from training.trainer import V3CellTrainer
from training.conditional_trainer import ConditionalTrainer

__all__ = [
    'GANLoss',
    'RelativisticAverageLSGAN',
    'compute_gradient_penalty_3d',
    'compute_gradient_penalty_2d',
    'compute_mutual_information_loss',
    'V3CellTrainer',
    'ConditionalTrainer',
]
