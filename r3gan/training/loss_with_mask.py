"""
Loss function with foreground mask support (for fine-tuning).

Extends the original R3GANLoss with two complementary mask constraints:

  1. Mask-weighted gradient penalty (R1/R2 regularisation):
       Gradients in the foreground region are penalised more heavily, making the
       discriminator's decision boundary steeper around the organoid.

  2. Masked adversarial loss branch:
       An additional adversarial loss is computed on foreground-only images
       (img × mask) and added to the standard full-image loss:
           L_total = L_adv_full + foreground_weight × L_adv_masked

Both G and D steps benefit from the masked branch.
Setting foreground_weight=0.0 recovers the original unmasked R3GAN training.
"""

import torch
from torch_utils import training_stats
from R3GAN.TrainerWithMask import AdversarialTrainingWithMask


class R3GANLossWithMask:
    """
    Extended R3GAN loss with foreground mask-weighted adversarial training.

    The foreground_weight parameter controls how much emphasis is placed on
    the organoid foreground vs the background:
      - foreground_weight=0.0  →  identical to standard R3GAN (no mask effect)
      - foreground_weight=1.0  →  masked branch has equal weight to full branch
      - foreground_weight=5.0  →  foreground quality is weighted 5× more
    """

    def __init__(self, G, D, augment_pipe=None, foreground_weight=5.0):
        """
        Args:
            G:                 Generator network.
            D:                 Discriminator network.
            augment_pipe:      ADA augmentation pipeline (optional).
            foreground_weight: Weighting factor for the foreground adversarial branch
                               and the gradient penalty. Default 5.0.
        """
        self.trainer = AdversarialTrainingWithMask(G, D)
        self.foreground_weight = foreground_weight

        if augment_pipe is not None:
            self.preprocessor = lambda x: augment_pipe(x.to(torch.float32)).to(x.dtype)
        else:
            self.preprocessor = lambda x: x

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gamma, gain, real_mask=None):
        """
        Accumulate gradients for one training phase.

        Args:
            phase:      'G' for generator update, 'D' for discriminator update.
            real_img:   Real images [B, C, H, W] in [-1, 1].
            real_c:     Class labels [B, c_dim].
            gen_z:      Generator noise [B, z_dim].
            gamma:      R1/R2 gradient penalty coefficient.
            gain:       Loss scale factor.
            real_mask:  Foreground masks [B, 1, H, W] in [0, 1], optional.
                        When provided, both adversarial loss and gradient penalty
                        are weighted to focus on the foreground region.
        """
        # --- Generator phase ---
        if phase == 'G':
            AdversarialLoss, RelativisticLogits = self.trainer.AccumulateGeneratorGradients(
                gen_z, real_img, real_c, gain, self.preprocessor,
                RealMasks=real_mask,
                ForegroundWeight=self.foreground_weight,
            )

            training_stats.report('Loss/scores/fake', RelativisticLogits)
            training_stats.report('Loss/signs/fake', RelativisticLogits.sign())
            training_stats.report('Loss/G/loss', AdversarialLoss)

        # --- Discriminator phase ---
        if phase == 'D':
            AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty = \
                self.trainer.AccumulateDiscriminatorGradients(
                    gen_z, real_img, real_c, gamma, gain, self.preprocessor,
                    RealMasks=real_mask,
                    ForegroundWeight=self.foreground_weight,
                )

            training_stats.report('Loss/scores/real', RelativisticLogits)
            training_stats.report('Loss/signs/real', RelativisticLogits.sign())
            training_stats.report('Loss/D/loss', AdversarialLoss)
            training_stats.report('Loss/r1_penalty', R1Penalty)
            training_stats.report('Loss/r2_penalty', R2Penalty)

            if real_mask is not None:
                training_stats.report('Loss/foreground_weight', self.foreground_weight)
                training_stats.report('Loss/mask_coverage', real_mask.float().mean())
