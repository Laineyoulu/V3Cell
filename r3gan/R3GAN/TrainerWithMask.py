"""
Trainer with foreground mask support (for fine-tuning).

Extends the original Trainer by adding two mask-based constraints:

  1. Mask-weighted gradient penalty (R1/R2):
       Foreground gradients are penalised more heavily, making the discriminator
       'landscape' steeper in the organoid region.

  2. Masked adversarial loss branch:
       An additional adversarial loss is computed on the foreground-masked images
       (img × mask) for both G and D.  The total loss becomes:
           L_total = L_adv_full + ForegroundWeight × L_adv_masked
       Setting ForegroundWeight=0.0 recovers the standard (unmasked) training.

Both constraints are tunable via the single ForegroundWeight scalar.
"""

import torch
import torch.nn as nn
from .Trainer import AdversarialTraining


class AdversarialTrainingWithMask(AdversarialTraining):
    """
    Extended Trainer that supports foreground mask-weighted losses.

    Two complementary mask constraints are applied:
      - Gradient penalty weighting: foreground regions penalised more during R1/R2.
      - Masked adversarial branch: separate adversarial loss on foreground-only images.
    """

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics, Masks=None, ForegroundWeight=5.0):
        """
        Zero-centred gradient penalty (R1/R2) with optional foreground weighting.

        Args:
            Samples:          [B, C, H, W] — input images with requires_grad=True.
            Critics:          [B] — discriminator scalar outputs.
            Masks:            [B, 1, H, W] — foreground masks in [0, 1], optional.
            ForegroundWeight: Gradient weight multiplier for foreground. Default 5.0.
                              background pixel weight = 1.0
                              foreground pixel weight = ForegroundWeight

        Returns:
            Penalty: [B] per-sample gradient penalty.
        """
        Gradient, = torch.autograd.grad(
            outputs=Critics.sum(), inputs=Samples, create_graph=True
        )

        GradientSquares = Gradient.square()  # [B, C, H, W]

        if Masks is not None:
            # W = 1 + Mask * (ForegroundWeight - 1)
            # background (Mask=0) → W=1,  foreground (Mask=1) → W=ForegroundWeight
            Weights = 1.0 + Masks * (ForegroundWeight - 1.0)  # [B, 1, H, W]
            GradientSquares = GradientSquares * Weights        # broadcast over channels

        return GradientSquares.sum([1, 2, 3])  # [B]

    @staticmethod
    def _apply_mask(Images, Masks):
        """
        Zero out background pixels.

        Multiplying by the mask in [-1, 1] space sets background to 0 (mid-grey),
        forcing the discriminator to judge only the foreground organoid region.

        Args:
            Images: [B, C, H, W] — images in [-1, 1].
            Masks:  [B, 1, H, W] — foreground masks in [0, 1].

        Returns:
            Masked images [B, C, H, W].
        """
        return Images * Masks  # broadcast mask over C channels

    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1,
                                     Preprocessor=lambda x: x,
                                     RealMasks=None, ForegroundWeight=5.0):
        """
        Accumulate generator gradients with optional foreground masked adversarial loss.

        The total generator loss is:
            L_G = L_G_full + ForegroundWeight * L_G_masked   (if RealMasks provided)
            L_G = L_G_full                                    (if RealMasks is None)

        Args:
            Noise:            [B, z_dim] — generator input noise.
            RealSamples:      [B, C, H, W] — real images (detached reference for relativistic loss).
            Conditions:       [B, c_dim] — class labels.
            Scale:            Loss scale factor.
            Preprocessor:     Augmentation pipeline (applied to both real and fake).
            RealMasks:        [B, 1, H, W] — foreground masks, optional.
            ForegroundWeight: Weight for the masked adversarial branch. Default 5.0.

        Returns:
            [AdversarialLoss, RelativisticLogits] (detached scalars/tensors).
        """
        FakeSamples = self.Generator(Noise, Conditions)
        RealSamples = RealSamples.detach()

        # --- Full-image adversarial loss (standard) ---
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)
        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)

        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        TotalGeneratorLoss = AdversarialLoss

        # --- Masked adversarial branch (foreground only) ---
        if RealMasks is not None and ForegroundWeight > 0.0:
            FakeMasked = self._apply_mask(FakeSamples, RealMasks)
            RealMasked = self._apply_mask(RealSamples, RealMasks)

            FakeLogitsMasked = self.Discriminator(Preprocessor(FakeMasked), Conditions)
            RealLogitsMasked = self.Discriminator(Preprocessor(RealMasked), Conditions)

            RelativisticLogitsMasked = FakeLogitsMasked - RealLogitsMasked
            AdversarialLossMasked = nn.functional.softplus(-RelativisticLogitsMasked)

            TotalGeneratorLoss = AdversarialLoss + ForegroundWeight * AdversarialLossMasked

        (Scale * TotalGeneratorLoss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]

    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1,
                                         Preprocessor=lambda x: x,
                                         RealMasks=None, ForegroundWeight=5.0):
        """
        Accumulate discriminator gradients with mask-weighted gradient penalty and
        optional masked adversarial branch.

        The total discriminator loss is:
            L_D = L_D_full + (Gamma/2) * (R1_weighted + R2)
                + ForegroundWeight * L_D_masked   (if RealMasks provided)

        Args:
            Noise:            [B, z_dim].
            RealSamples:      [B, C, H, W].
            Conditions:       [B, c_dim].
            Gamma:            R1/R2 gradient penalty coefficient.
            Scale:            Loss scale factor.
            Preprocessor:     Augmentation pipeline.
            RealMasks:        [B, 1, H, W] — foreground masks, optional.
            ForegroundWeight: Weight for masked branch and foreground gradient penalty.

        Returns:
            [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty] (detached).
        """
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = self.Generator(Noise, Conditions).detach().requires_grad_(True)

        RealLogits = self.Discriminator(Preprocessor(RealSamples), Conditions)
        FakeLogits = self.Discriminator(Preprocessor(FakeSamples), Conditions)

        # Mask-weighted gradient penalty: foreground gets higher penalty weight
        R1Penalty = self.ZeroCenteredGradientPenalty(
            RealSamples, RealLogits, RealMasks, ForegroundWeight
        )
        R2Penalty = self.ZeroCenteredGradientPenalty(
            FakeSamples, FakeLogits, None, ForegroundWeight
        )

        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)

        # --- Masked adversarial branch (foreground only) ---
        if RealMasks is not None and ForegroundWeight > 0.0:
            # Detach and re-enable grad for masked samples
            RealMasked = self._apply_mask(RealSamples.detach(), RealMasks).requires_grad_(True)
            FakeMasked = self._apply_mask(FakeSamples.detach(), RealMasks).requires_grad_(True)

            RealLogitsMasked = self.Discriminator(Preprocessor(RealMasked), Conditions)
            FakeLogitsMasked = self.Discriminator(Preprocessor(FakeMasked), Conditions)

            RelativisticLogitsMasked = RealLogitsMasked - FakeLogitsMasked
            AdversarialLossMasked = nn.functional.softplus(-RelativisticLogitsMasked)

            # Masked branch gradient penalty (no extra fg weighting needed here)
            R1Masked = self.ZeroCenteredGradientPenalty(RealMasked, RealLogitsMasked)
            R2Masked = self.ZeroCenteredGradientPenalty(FakeMasked, FakeLogitsMasked)

            MaskedBranchLoss = (
                AdversarialLossMasked + (Gamma / 2) * (R1Masked + R2Masked)
            )
            DiscriminatorLoss = DiscriminatorLoss + ForegroundWeight * MaskedBranchLoss

        (Scale * DiscriminatorLoss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
