# -*- coding: utf-8 -*-
"""
Loss functions for V3Cell (MoCoGAN-HD specification).

Contents:
  - GANLoss:                        Basic LSGAN or vanilla GAN loss.
  - RelativisticAverageLSGAN:       RaLSGAN — relativistic average variant.
  - compute_gradient_penalty_3d/2d: WGAN-GP gradient penalty for 3D/2D discriminators.
  - compute_mutual_information_loss: MI loss to prevent motion mode collapse.
  - TemporalConsistencyLoss:        Penalise abrupt inter-frame changes.
  - compute_temporal_direction_loss: Cosine-similarity loss on frame-difference directions.
  - compute_progressive_recon_loss: Decaying-weight L1 reconstruction loss.
  - compute_feature_matching_loss:  Discriminator feature matching (MoCoGAN-HD).
  - ContrastiveEncoder / MemoryBank / compute_contrastive_loss*:
                                    Contrastive identity-preservation losses (MoCoGAN-HD F network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class GANLoss(nn.Module):
    """
    Basic GAN loss supporting LSGAN (MSE) and vanilla GAN (BCE).

    Handles both single-tensor and multi-scale list outputs from discriminators.
    """

    def __init__(
        self,
        use_lsgan: bool = True,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create a target tensor of the correct shape filled with the appropriate label."""
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.numel() != input.numel():
                self.real_label_var = torch.full_like(input, self.real_label).detach()
            return self.real_label_var.to(input.device)
        else:
            if self.fake_label_var is None or self.fake_label_var.numel() != input.numel():
                self.fake_label_var = torch.full_like(input, self.fake_label).detach()
            return self.fake_label_var.to(input.device)

    def forward(
        self,
        input: Union[torch.Tensor, List],
        target_is_real: bool,
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            input:          Discriminator output — tensor or list-of-lists (multi-scale).
            target_is_real: Whether input should be treated as real.

        Returns:
            loss: Scalar GAN loss.
        """
        if isinstance(input, list) and isinstance(input[0], list):
            loss = sum(
                self.loss(inp[-1], self.get_target_tensor(inp[-1], target_is_real))
                for inp in input
            )
            return loss
        elif isinstance(input, list):
            pred = input[-1]
            return self.loss(pred, self.get_target_tensor(pred, target_is_real))
        else:
            return self.loss(input, self.get_target_tensor(input, target_is_real))


class RelativisticAverageLSGAN(GANLoss):
    """
    Relativistic Average LSGAN (RaLSGAN).

    Discriminator update (real is input_1, fake is input_2):
        L_D = E[(D(x_real) − E[D(x_fake)] − 1)²]
            + E[(D(x_fake) − E[D(x_real)] + 1)²]

    Generator update (fake is input_1, real is input_2):
        L_G = E[(D(x_real) − E[D(x_fake)] + 1)²]
            + E[(D(x_fake) − E[D(x_real)] − 1)²]

    Call with target_is_real=True  when input_1 should score high (D update, real batch).
    Call with target_is_real=False when input_1 should score low  (D update, fake batch).
    The G update logic is obtained by swapping real/fake and flipping target_is_real.
    """

    def forward(
        self,
        input_1: Union[torch.Tensor, List],
        input_2: Union[torch.Tensor, List],
        target_is_real: bool,
    ) -> torch.Tensor:
        """
        Args:
            input_1:        Discriminator output for the "target" batch.
            input_2:        Discriminator output for the "reference" batch.
            target_is_real: Whether input_1 is the real batch.

        Returns:
            loss: Scalar relativistic average LSGAN loss.
        """
        if isinstance(input_1, list) and len(input_1) > 0 and isinstance(input_1[0], list):
            loss = 0.0
            for inp1, inp2 in zip(input_1, input_2):
                pred, ref = inp1[-1], inp2[-1]
                loss += self.loss(pred - torch.mean(ref), self.get_target_tensor(pred, target_is_real))
            return loss
        elif isinstance(input_1, list):
            pred = input_1[-1] if input_1 else input_1
            ref = input_2[-1] if isinstance(input_2, list) and input_2 else input_2
            return self.loss(pred - torch.mean(ref), self.get_target_tensor(pred, target_is_real))
        else:
            return self.loss(
                input_1 - torch.mean(input_2),
                self.get_target_tensor(input_1, target_is_real),
            )


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DataParallel wrapper to access the underlying module."""
    return model.module if hasattr(model, 'module') else model


def compute_gradient_penalty_3d(
    real_video: torch.Tensor,
    fake_video: torch.Tensor,
    discriminator: nn.Module,
    num_D: int = 2,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty for the 3D VideoDiscriminator.

    Computes the penalty on interpolated samples between real and fake videos.
    The discriminator is unwrapped from any DataParallel wrapper before calling
    autograd.grad to avoid computation-graph breakage across devices.

    Args:
        real_video:    [B, C, T, H, W] — real video batch.
        fake_video:    [B, C, T, H, W] — generated video batch.
        discriminator: 3D discriminator (may be DataParallel-wrapped).
        num_D:         Number of discriminator scales.
        lambda_gp:     Gradient penalty coefficient.

    Returns:
        gradient_penalty: Scalar gradient penalty loss.
    """
    disc = _unwrap_model(discriminator)
    batch_size = real_video.size(0)
    device = real_video.device

    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device).expand_as(real_video)
    interpolates = (alpha * real_video.data + (1 - alpha) * fake_video.data).clone()
    interpolates.requires_grad_(True)

    pred_interpolates = disc(interpolates)
    gradient_penalty = 0.0

    if isinstance(pred_interpolates, list):
        for cur_pred in pred_interpolates:
            output = cur_pred[-1] if isinstance(cur_pred, list) else cur_pred
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=interpolates,
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(batch_size, -1)
            gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty = (gradient_penalty / num_D) * lambda_gp
    else:
        gradients = torch.autograd.grad(
            outputs=pred_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(pred_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

    return gradient_penalty


def compute_gradient_penalty_2d(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    discriminator: nn.Module,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty for the 2D ImageDiscriminator.

    Args:
        real_images:   [B, C, H, W] — real image batch.
        fake_images:   [B, C, H, W] — generated image batch.
        discriminator: 2D discriminator (may be DataParallel-wrapped).
        lambda_gp:     Gradient penalty coefficient.

    Returns:
        gradient_penalty: Scalar gradient penalty loss.
    """
    disc = _unwrap_model(discriminator)
    batch_size = real_images.size(0)
    device = real_images.device

    alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_images)
    interpolates = (alpha * real_images.data + (1 - alpha) * fake_images.data).clone()
    interpolates.requires_grad_(True)

    pred_interpolates = disc(interpolates)
    gradients = torch.autograd.grad(
        outputs=pred_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(pred_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp


def compute_mutual_information_loss(
    rand_in: torch.Tensor,
    rand_rec: torch.Tensor,
) -> torch.Tensor:
    """
    Mutual-information loss for noise injection (MoCoGAN-HD).

    Encourages the LSTM to use the injected noise eps_t by maximising the
    mutual information I(h_t; eps_t).  In practice this is implemented as a
    cosine-similarity loss between the injected noise and its reconstruction
    from the LSTM hidden state.

    High MI → LSTM leverages noise → diverse motion trajectories (avoids collapse).

    Args:
        rand_in:  [B, T, z_dim] — injected noise vectors.
        rand_rec: [B, T, z_dim] — noise reconstructed from LSTM hidden state.

    Returns:
        loss: Scalar negative cosine similarity (lower = more aligned).
    """
    if rand_in is None or rand_rec is None:
        return torch.tensor(0.0)

    if rand_in.dim() == 3:
        B, T, z_dim = rand_in.shape
        rand_in = rand_in.reshape(B * T, z_dim)
        rand_rec = rand_rec.reshape(B * T, z_dim)

    return -torch.mean(F.cosine_similarity(rand_rec, rand_in.detach()))


class TemporalConsistencyLoss(nn.Module):
    """
    Penalise abrupt changes between consecutive frames.

    Computed as the mean squared difference between adjacent frames.  Can be
    used as a soft regulariser to encourage smooth developmental trajectories.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W]

        Returns:
            loss: Scalar temporal consistency loss.
        """
        if video.shape[1] < 2:
            return torch.tensor(0.0, device=video.device)
        frame_diff = video[:, 1:] - video[:, :-1]
        return self.weight * frame_diff.pow(2).mean()


def compute_temporal_direction_loss(
    pred_video: torch.Tensor,
    real_video: torch.Tensor,
) -> torch.Tensor:
    """
    Temporal direction consistency loss.

    Penalises predictions whose frame-to-frame velocity vectors point in the
    opposite direction to those of the real video.  Prevents the model from
    generating time-reversed developmental sequences.

    Method:
      1. Compute velocity (frame difference) for both real and predicted videos.
      2. Measure cosine similarity between corresponding velocity vectors.
      3. Loss = mean(1 - cos_sim), so same direction → 0, opposite → 2.

    Args:
        pred_video: [B, T, C, H, W] — predicted video.
        real_video: [B, T, C, H, W] — real video.

    Returns:
        loss: Scalar temporal direction loss.
    """
    if pred_video.shape[1] < 3:
        return torch.tensor(0.0, device=pred_video.device)

    pred_diff = pred_video[:, 1:] - pred_video[:, :-1]  # [B, T-1, C, H, W]
    real_diff = real_video[:, 1:] - real_video[:, :-1]

    B, T_1, C, H, W = pred_diff.shape
    pred_flat = pred_diff.view(B, T_1, -1)
    real_flat = real_diff.view(B, T_1, -1)

    cos_sim = F.cosine_similarity(pred_flat, real_flat, dim=-1)  # [B, T-1]
    return (1 - cos_sim).mean()


def compute_progressive_recon_loss(
    pred_video: torch.Tensor,
    real_video: torch.Tensor,
    n_condition: int,
    decay: float = 0.95,
) -> torch.Tensor:
    """
    Progressive reconstruction loss with exponentially decaying frame weights.

    Assigns weight 1 to condition frames and weight decay^(t - n_condition) to
    predicted frames.  This reduces the penalty for errors in later frames
    where prediction uncertainty is naturally higher, mitigating the adverse
    effect of error accumulation on gradient estimates.

    Args:
        pred_video:  [B, T, C, H, W] — predicted full video.
        real_video:  [B, T, C, H, W] — real full video.
        n_condition: Number of condition (observed) frames.
        decay:       Per-step weight decay factor for predicted frames.

    Returns:
        loss: Weighted L1 reconstruction loss.
    """
    B, T, C, H, W = pred_video.shape
    device = pred_video.device

    weights = torch.ones(T, device=device)
    for t in range(n_condition, T):
        weights[t] = decay ** (t - n_condition)
    weights = weights / weights.sum() * T  # normalise so total weight is T

    frame_losses = F.l1_loss(pred_video, real_video, reduction='none')
    frame_losses = frame_losses.mean(dim=(2, 3, 4))  # [B, T]

    return (frame_losses * weights.unsqueeze(0)).mean()


def compute_feature_matching_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
    num_D: int = 2,
) -> torch.Tensor:
    """
    Feature matching loss (MoCoGAN-HD style) for the 3D discriminator.

    Matches intermediate feature maps between real and generated samples.
    This gives the generator a richer training signal and stabilises GAN
    training by preventing the discriminator from becoming too dominant.

    Args:
        real_features: List[List[Tensor]] — [num_D][num_layers] real features.
        fake_features: List[List[Tensor]] — [num_D][num_layers] fake features.
        num_D:         Number of discriminator scales.

    Returns:
        loss: Scalar feature matching loss (L1, averaged over scales).
    """
    device = None
    for scale in real_features:
        for t in scale:
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is not None:
            break
    loss = torch.zeros(1, device=device)
    for i in range(num_D):
        for j in range(len(real_features[i]) - 1):  # exclude the final output layer
            real_feat = real_features[i][j].detach()  # do not backprop through real
            fake_feat = fake_features[i][j]
            loss = loss + F.l1_loss(fake_feat, real_feat)
    return loss / num_D


def compute_feature_matching_loss_2d(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
) -> torch.Tensor:
    """
    Simplified feature matching loss for the single-scale 2D discriminator.

    Args:
        real_features: Discriminator output / features for real samples.
        fake_features: Discriminator output / features for fake samples.

    Returns:
        loss: Scalar L1 loss.
    """
    return F.l1_loss(fake_features, real_features.detach())


# =============================================================================
# Contrastive Learning (MoCoGAN-HD F network)
# =============================================================================

class ContrastiveEncoder(nn.Module):
    """
    Identity feature encoder for contrastive learning (MoCoGAN-HD F network).

    Extracts appearance features from individual frames for the contrastive
    identity-preservation loss:
      - Positive pairs: the first real frame and any generated frame from the
                        same video (same organoid identity).
      - Negative pairs: frames from different videos (different identities).

    Features are L2-normalised to unit sphere so that cosine similarity equals
    the dot product used in InfoNCE.
    """

    def __init__(
        self,
        img_channels: int = 3,
        img_resolution: int = 256,
        feature_dim: int = 128,
        base_channels: int = 64,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Lightweight CNN backbone: 256 → 128 → 64 → 32 → 16 → 8 → pool
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # MLP projection head.
        self.projector = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 4, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] — input frame.

        Returns:
            features: [B, feature_dim] — L2-normalised feature vectors.
        """
        h = self.encoder(x).view(x.size(0), -1)
        return F.normalize(self.projector(h), dim=-1)


def compute_contrastive_loss(
    first_frame_features: torch.Tensor,
    other_frame_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss using in-batch negatives only.

    Positive pairs: (first_frame[i], other_frame[i]) — same organoid, different time.
    Negative pairs: (first_frame[i], other_frame[j≠i]) — different organoids.

    Args:
        first_frame_features: [B, feature_dim] — anchor (first real frame).
        other_frame_features: [B, feature_dim] — positive (generated frame, same video).
        temperature:           Softmax temperature τ.

    Returns:
        loss: Scalar InfoNCE / NT-Xent loss.
    """
    batch_size = first_frame_features.shape[0]
    device = first_frame_features.device

    # Similarity matrix: (i, j) = cos_sim(first[i], other[j]) / τ
    sim_matrix = torch.matmul(first_frame_features, other_frame_features.T) / temperature

    # Diagonal elements are the positive pairs.
    labels = torch.arange(batch_size, device=device)
    return F.cross_entropy(sim_matrix, labels)


def compute_contrastive_loss_with_memory(
    query_features: torch.Tensor,
    positive_features: torch.Tensor,
    memory_bank: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss with a memory bank of additional negative samples.

    Using a memory bank allows a much larger effective negative-sample set than
    is possible with small batch sizes, stabilising the contrastive signal.

    Args:
        query_features:    [B, feature_dim] — anchor features (first real frame).
        positive_features: [B, feature_dim] — positive features (generated frame).
        memory_bank:       [K, feature_dim] — stored historical features as negatives.
        temperature:        Softmax temperature τ.

    Returns:
        loss: Scalar InfoNCE loss with K+1 contrastive classes.
    """
    batch_size = query_features.shape[0]
    device = query_features.device

    # Positive similarity [B, 1]
    pos_sim = torch.sum(query_features * positive_features, dim=-1, keepdim=True) / temperature

    # Negative similarity against memory bank [B, K]
    neg_sim = torch.matmul(query_features, memory_bank.T) / temperature

    # Logit matrix [B, 1+K] — positive is always at index 0
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


class MemoryBank:
    """
    FIFO memory bank for contrastive learning negative samples.

    Stores L2-normalised feature vectors from recent training batches.
    Updated by enqueue/dequeue so that the oldest features are replaced first.
    """

    def __init__(self, size: int = 4096, feature_dim: int = 128):
        self.size = size
        self.feature_dim = feature_dim
        self.bank: torch.Tensor = None
        self.ptr = 0

    def update(self, features: torch.Tensor):
        """
        Add a new batch of features to the memory bank (overwrites oldest entries).

        Args:
            features: [B, feature_dim] — new feature vectors (detached).
        """
        batch_size = features.shape[0]
        features = features.detach()

        if self.bank is None:
            self.bank = torch.zeros(self.size, self.feature_dim, device=features.device)

        end = self.ptr + batch_size
        if end <= self.size:
            self.bank[self.ptr:end] = features
        else:
            # Wrap around: write the overflow to the beginning.
            overflow = end - self.size
            self.bank[self.ptr:] = features[:self.size - self.ptr]
            self.bank[:overflow] = features[self.size - self.ptr:]

        self.ptr = (self.ptr + batch_size) % self.size

    def get(self) -> torch.Tensor:
        """
        Return a detached clone of all stored features.

        Returns None if the bank has not been populated yet.
        """
        if self.bank is None:
            return None
        return self.bank.clone().detach()
