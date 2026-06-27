# -*- coding: utf-8 -*-
"""
Encoder networks for V3Cell.

Encode observed organoid frames into the R3GAN latent space so that the
MotionRNN can be conditioned on real developmental history.

Design choices:
  - ImageEncoder:   Per-frame VAE-style encoder.  Uses the reparameterisation
                    trick so that sampled latents stay close to the N(0,1) prior
                    that R3GAN was trained with.
  - SequenceEncoder: Wraps ImageEncoder with a temporal LSTM that aggregates
                    frame-level latents into an initial hidden state (h_k, c_k)
                    for MotionRNN.  This allows the motion predictor to benefit
                    from the full observed prefix rather than just the last frame.
  - compute_kl_loss: Closed-form KL divergence KL(q(z|x) || N(0,1)) used to
                    regularise the encoder distribution during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class GradualStyleBlock(nn.Module):
    """
    Multi-scale style feature extractor (pSp-inspired).

    Extracts a spatial feature map, pools it to a fixed size, and projects
    it to the target w_dim via a linear layer.
    """

    def __init__(self, in_c: int, out_c: int, spatial_size: int):
        super().__init__()
        self.out_c = out_c
        self.spatial_size = spatial_size

        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))
        self.fc = nn.Linear(out_c * spatial_size * spatial_size, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv(x), 0.2)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class ResBlock(nn.Module):
    """Standard pre-activation residual block with optional spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + self.shortcut(x), 0.2)


class ImageEncoder(nn.Module):
    """
    Single-frame VAE-style encoder.

    Maps a 256×256 RGB image to a latent vector in the R3GAN noise space
    (z_dim = 64 by default).  Using the reparameterisation trick keeps sampled
    latents within the N(0,1) prior that R3GAN was trained with, ensuring that
    the frozen decoder produces valid images.

    Architecture:
        Input (3×256×256)
          → stem conv (→ 128×128)
          → ResBlock ×5 (progressive downsampling to 4×4)
          → global average pool → flatten
          → fc_mu / fc_logvar → z via reparameterisation
    """

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        w_dim: int = 64,           # Must match R3GAN NoiseDimension
        base_channels: int = 64,
        use_vae: bool = True,
    ):
        """
        Args:
            img_resolution: Spatial resolution of input images.
            img_channels:   Number of input image channels.
            w_dim:          Output latent dimension (must match R3GAN's NoiseDimension).
            base_channels:  Base width of the ResNet backbone.
            use_vae:        If True, output μ and log σ² and apply reparameterisation.
                            If False, output a deterministic latent (for inference only).
        """
        super().__init__()

        self.img_resolution = img_resolution
        self.w_dim = w_dim
        self.use_vae = use_vae

        # Stem: 256 → 128
        self.input_layer = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 7, 2, 3),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
        )

        # ResNet backbone: 128 → 64 → 32 → 16 → 8 → 4
        self.encoder = nn.Sequential(
            ResBlock(base_channels,     base_channels * 2, downsample=True),
            ResBlock(base_channels * 2, base_channels * 4, downsample=True),
            ResBlock(base_channels * 4, base_channels * 8, downsample=True),
            ResBlock(base_channels * 8, base_channels * 8, downsample=True),
            ResBlock(base_channels * 8, base_channels * 8, downsample=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        if use_vae:
            self.fc_mu = nn.Sequential(
                nn.Linear(base_channels * 8, w_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(w_dim * 2, w_dim),
            )
            self.fc_logvar = nn.Sequential(
                nn.Linear(base_channels * 8, w_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(w_dim * 2, w_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(base_channels * 8, w_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(w_dim, w_dim),
            )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick: z = μ + σ * ε,  ε ~ N(0, I).

        During inference the mean is returned directly (no sampling noise).

        Args:
            mu:      [B, w_dim] — posterior mean.
            log_var: [B, w_dim] — log posterior variance (clamped for stability).

        Returns:
            z: [B, w_dim]
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(
        self,
        x: torch.Tensor,
        return_distribution: bool = False,
    ):
        """
        Encode a single image to the R3GAN latent space.

        Args:
            x:                   [B, C, H, W] — input images in [-1, 1].
            return_distribution: If True, also return μ and log σ² (for KL loss).

        Returns:
            w:       [B, w_dim] — encoded latent.
            mu:      [B, w_dim] — VAE mean       (only if return_distribution=True).
            log_var: [B, w_dim] — VAE log-variance (only if return_distribution=True).
        """
        x = self.input_layer(x)
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)

        if self.use_vae:
            mu = self.fc_mu(x)
            log_var = torch.clamp(self.fc_logvar(x), min=-10, max=10)
            w = self.reparameterize(mu, log_var)
            if return_distribution:
                return w, mu, log_var
            return w
        else:
            w = self.fc(x)
            if return_distribution:
                return w, None, None
            return w


def compute_kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from the VAE posterior to the standard normal prior.

        KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log σ² - μ² - σ²)

    Minimising this term encourages the encoder distribution to stay close to
    N(0, 1), which is compatible with the R3GAN prior.

    Args:
        mu:      [B, z_dim] — VAE posterior mean.
        log_var: [B, z_dim] — VAE posterior log-variance.

    Returns:
        kl_loss: Scalar KL divergence (mean over batch and dimension).
    """
    if mu is None or log_var is None:
        return torch.tensor(0.0)
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


class SequenceEncoder(nn.Module):
    """
    Sequence encoder for the V3Cell conditional generation pipeline.

    Encodes a temporal sequence of early frames into:
      1. A per-frame latent sequence z_seq  [B, T, w_dim] — fed directly to
         MotionRNN as the observed prefix.
      2. An initial LSTM hidden state (h_k, c_k) — used to warm-start MotionRNN
         so that future predictions are coherent with the observed past.

    Internally uses an ImageEncoder (VAE-style) followed by a unidirectional LSTM
    that aggregates frame-level latents temporally.
    """

    def __init__(
        self,
        img_resolution: int = 256,
        img_channels: int = 3,
        w_dim: int = 64,
        hidden_dim: int = 384,    # Matches MotionRNN hidden_dim
        num_layers: int = 2,
        base_channels: int = 64,
        use_vae: bool = True,
    ):
        """
        Args:
            img_resolution: Input image spatial resolution.
            img_channels:   Number of image channels.
            w_dim:          Latent dimension (must equal MotionRNN z_dim).
            hidden_dim:     LSTM hidden state size (must equal MotionRNN hidden_dim).
            num_layers:     Number of LSTM layers (must equal MotionRNN num_layers).
            base_channels:  Base width of the per-frame ResNet encoder.
            use_vae:        Whether to apply the VAE reparameterisation trick.
        """
        super().__init__()

        self.w_dim = w_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_vae = use_vae

        # Per-frame VAE encoder.
        self.image_encoder = ImageEncoder(
            img_resolution=img_resolution,
            img_channels=img_channels,
            w_dim=w_dim,
            base_channels=base_channels,
            use_vae=use_vae,
        )

        # Temporal LSTM: aggregates frame-level latents into a hidden state.
        # The final hidden state is projected to initialise MotionRNN.
        self.sequence_lstm = nn.LSTM(
            input_size=w_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Linear projections to convert this encoder's hidden state into the
        # format expected by MotionRNN's LSTM.
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cell_proj = nn.Linear(hidden_dim, hidden_dim)

    def encode_frames(
        self,
        frames: torch.Tensor,
        return_distribution: bool = False,
    ):
        """
        Encode each frame independently to produce a per-frame latent sequence.

        Args:
            frames:              [B, T, C, H, W] — frame sequence.
            return_distribution: If True, also return per-frame μ and log σ².

        Returns:
            w_seq:      [B, T, w_dim]
            mu_seq:     [B, T, w_dim]   (only if return_distribution=True)
            logvar_seq: [B, T, w_dim]   (only if return_distribution=True)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        if return_distribution and self.use_vae:
            w_flat, mu_flat, logvar_flat = self.image_encoder(
                frames_flat, return_distribution=True
            )
            return (
                w_flat.reshape(B, T, self.w_dim),
                mu_flat.reshape(B, T, self.w_dim),
                logvar_flat.reshape(B, T, self.w_dim),
            )
        else:
            w_flat = self.image_encoder(frames_flat)
            return w_flat.reshape(B, T, self.w_dim)

    def forward(
        self,
        frames: torch.Tensor,
        return_distribution: bool = False,
    ):
        """
        Encode an early-frame sequence and produce the MotionRNN warm-start state.

        Args:
            frames:              [B, T, C, H, W] — observed early frames in [-1, 1].
            return_distribution: If True, also return per-frame μ and log σ².

        Returns:
            w_seq:   [B, T, w_dim]   — per-frame latent sequence.
            (h_k, c_k):              — LSTM hidden/cell states for MotionRNN.
            mu_seq:  [B, T, w_dim]   — VAE means       (optional).
            logvar_seq: [B, T, w_dim]— VAE log-variances (optional).
        """
        if return_distribution and self.use_vae:
            w_seq, mu_seq, logvar_seq = self.encode_frames(
                frames, return_distribution=True
            )
        else:
            w_seq = self.encode_frames(frames)
            mu_seq, logvar_seq = None, None

        # Run the temporal LSTM over the per-frame latents.
        _, (h_n, c_n) = self.sequence_lstm(w_seq)

        # Project the final hidden state to initialise MotionRNN.
        # h_n: [num_layers, B, hidden_dim]
        h_k = self.hidden_proj(h_n)
        c_k = self.cell_proj(c_n)

        if return_distribution:
            return w_seq, (h_k, c_k), mu_seq, logvar_seq
        return w_seq, (h_k, c_k)


def test_encoder():
    """Quick smoke test for SequenceEncoder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = SequenceEncoder(
        img_resolution=256,
        img_channels=3,
        w_dim=512,
        hidden_dim=256,
        num_layers=2,
    ).to(device)

    batch_size, n_frames = 4, 10
    frames = torch.randn(batch_size, n_frames, 3, 256, 256).to(device)
    w_seq, (h_k, c_k) = encoder(frames)

    assert w_seq.shape == (batch_size, n_frames, 512)
    assert h_k.shape == (2, batch_size, 256)
    assert c_k.shape == (2, batch_size, 256)
    print("SequenceEncoder test passed.")


if __name__ == '__main__':
    test_encoder()
