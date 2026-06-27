# -*- coding: utf-8 -*-
"""
ConditionalTrainer: Conditional video generation trainer for V3Cell.

Training objective:
  Given the first k frames of a real organoid development video, predict
  the full video trajectory and match it against the ground truth.

Training pipeline:
  1. Sample a real video [B, T, C, H, W] from the dataloader.
  2. Take the first k frames as condition (k is optionally randomised).
  3. SequenceEncoder encodes condition frames → latent sequence + LSTM warm-start.
  4. MotionRNN predicts future latents autoregressively.
  5. Frozen R3GAN decodes every latent to a frame.
  6. Compute composite loss:
       L = λ_recon * L_recon          (progressive L1 reconstruction)
         + λ_adv   * (L_2d + L_3d)   (Relativistic LSGAN adversarial)
         + λ_motion* L_motion         (motion smoothness)
         + λ_temp  * L_temporal       (temporal direction consistency)
         + λ_mutual* L_mutual         (MI loss — prevents motion collapse)
         + λ_feat  * L_feat           (feature matching — MoCoGAN-HD)
         + λ_kl    * L_KL             (VAE KL regularisation)
         + λ_contra* L_contrastive    (identity preservation — MoCoGAN-HD F net)
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path as _Path
_project_root = _Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models import V3CellGenerator, ImageDiscriminator, VideoDiscriminator
from models.discriminators import prepare_image_disc_input, prepare_video_disc_input
from training.losses import (
    RelativisticAverageLSGAN,
    compute_gradient_penalty_3d,
    compute_temporal_direction_loss,
    compute_progressive_recon_loss,
    compute_mutual_information_loss,
    compute_feature_matching_loss,
    ContrastiveEncoder,
    MemoryBank,
    compute_contrastive_loss,
    compute_contrastive_loss_with_memory,
)
from models.encoder import compute_kl_loss


def flip_video(x: torch.Tensor) -> torch.Tensor:
    """Randomly flip a video tensor horizontally (data augmentation)."""
    if random.random() > 0.5:
        return torch.flip(x, [-1])
    return x


class ConditionalTrainer:
    """
    Conditional V3Cell trainer — predicts full development from early frames.

    This is the primary trainer used in the V3Cell paper.  It conditions on
    the first k observed frames to predict the complete organoid developmental
    trajectory.

    The number of condition frames k can be fixed or sampled uniformly from a
    range at each training step, which improves robustness to variable input
    lengths at inference time.

    Attributes:
        generator:          V3CellGenerator (Encoder + MotionRNN + frozen R3GAN).
        image_disc:         2D frame-pair discriminator.
        video_disc:         3D multi-scale video discriminator.
        opt_generator:      Adam optimiser for Encoder + MotionRNN (+ ContrastiveEncoder).
                            Must be assigned externally before calling train():
                            ``trainer.opt_generator = torch.optim.Adam(...)``.
    """

    def __init__(
        self,
        generator: V3CellGenerator,
        image_discriminator: ImageDiscriminator,
        video_discriminator: VideoDiscriminator,
        dataloader: DataLoader,
        device: torch.device,
        # --- Temporal setup ---
        n_frames: int = 64,
        n_condition_frames: int = 10,
        condition_frames_range: Tuple[int, int] = (5, 15),
        random_condition: bool = True,
        # --- Training schedule ---
        G_step: int = 1,
        num_D: int = 2,
        # --- Loss weights ---
        lambda_recon: float = 10.0,      # Reconstruction loss
        lambda_adv: float = 1.0,         # Adversarial loss (2D + 3D combined)
        lambda_motion: float = 0.1,      # Motion smoothness
        lambda_temporal: float = 1.0,    # Temporal direction consistency
        lambda_mutual: float = 1.0,      # MI loss (prevents motion mode collapse)
        lambda_feat: float = 10.0,       # Feature matching (MoCoGAN-HD)
        lambda_kl: float = 0.01,         # KL divergence (VAE regularisation)
        lambda_contrastive: float = 1.0, # Contrastive identity preservation
        # --- Advanced options ---
        use_progressive_recon: bool = True,
        progressive_decay: float = 0.95,
        use_feature_matching: bool = True,
        use_kl_loss: bool = True,
        use_contrastive: bool = True,
        contrastive_feature_dim: int = 128,
        memory_bank_size: int = 4096,
        # --- Output ---
        output_dir: Union[str, Path] = 'outputs',
    ):
        self.generator = generator.to(device)
        self.image_disc = image_discriminator.to(device)
        self.video_disc = video_discriminator.to(device)
        self.dataloader = dataloader
        self.device = device

        self.n_frames = n_frames
        self.n_condition_frames = n_condition_frames
        self.condition_frames_range = condition_frames_range
        self.random_condition = random_condition
        self.n_G_steps = G_step
        self.num_D = num_D

        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_motion = lambda_motion
        self.lambda_temporal = lambda_temporal
        self.lambda_mutual = lambda_mutual
        self.lambda_feat = lambda_feat
        self.lambda_kl = lambda_kl
        self.lambda_contrastive = lambda_contrastive

        self.use_progressive_recon = use_progressive_recon
        self.progressive_decay = progressive_decay
        self.use_feature_matching = use_feature_matching
        self.use_kl_loss = use_kl_loss
        self.use_contrastive = use_contrastive

        # Contrastive encoder F and memory bank (MoCoGAN-HD).
        if use_contrastive:
            gen_module = generator.module if hasattr(generator, 'module') else generator
            self.contrastive_encoder = ContrastiveEncoder(
                img_channels=3,
                img_resolution=gen_module.img_resolution,
                feature_dim=contrastive_feature_dim,
            ).to(device)
            self.memory_bank = MemoryBank(size=memory_bank_size, feature_dim=contrastive_feature_dim)
        else:
            self.contrastive_encoder = None
            self.memory_bank = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion_gan = RelativisticAverageLSGAN()

        self.cur_step = 0
        self.cur_kimg = 0.0
        self.stats = defaultdict(list)

        # Logging infrastructure.
        self.log_file = self.output_dir / 'training_log.csv'
        self.config_file = self.output_dir / 'config.json'
        self.tensorboard_dir = self.output_dir / 'tensorboard'
        self.writer = None  # TensorBoard SummaryWriter (lazy-initialised)

        self._save_config()

    # ------------------------------------------------------------------
    # Configuration and logging
    # ------------------------------------------------------------------

    def _save_config(self):
        """Persist hyperparameters to a JSON file in the output directory."""
        import json
        config = {
            'n_frames': self.n_frames,
            'n_condition_frames': self.n_condition_frames,
            'condition_frames_range': self.condition_frames_range,
            'random_condition': self.random_condition,
            'n_G_steps': self.n_G_steps,
            'num_D': self.num_D,
            'lambda_recon': self.lambda_recon,
            'lambda_adv': self.lambda_adv,
            'lambda_motion': self.lambda_motion,
            'lambda_temporal': self.lambda_temporal,
            'lambda_mutual': self.lambda_mutual,
            'lambda_feat': self.lambda_feat,
            'lambda_kl': self.lambda_kl,
            'lambda_contrastive': self.lambda_contrastive,
            'use_progressive_recon': self.use_progressive_recon,
            'progressive_decay': self.progressive_decay,
            'use_feature_matching': self.use_feature_matching,
            'use_kl_loss': self.use_kl_loss,
            'use_contrastive': self.use_contrastive,
            'output_dir': str(self.output_dir),
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def _init_tensorboard(self):
        """Lazily initialise a TensorBoard SummaryWriter."""
        if self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(self.tensorboard_dir))
            except ImportError:
                print("TensorBoard not available — skipping.")
                self.writer = False  # sentinel: do not retry

    def _log_to_csv(self, step: int, kimg: float, losses: Dict[str, float]):
        """Append a row of loss values to the CSV training log."""
        import csv
        file_exists = self.log_file.exists()
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['step', 'kimg'] + list(losses.keys()))
            writer.writerow([step, f'{kimg:.2f}'] + [f'{v:.6f}' for v in losses.values()])

    def _log_to_tensorboard(self, step: int, losses: Dict[str, float]):
        """Write scalar loss values to TensorBoard."""
        if self.writer is None:
            self._init_tensorboard()
        if self.writer:
            for name, value in losses.items():
                self.writer.add_scalar(f'loss/{name}', value, step)

    def _save_final_stats(self):
        """Write a summary of final training statistics to JSON."""
        import json
        summary = {
            'total_steps': self.cur_step,
            'total_kimg': self.cur_kimg,
            'final_losses': {},
            'best_losses': {},
        }
        for name, values in self.stats.items():
            if values:
                summary['final_losses'][name] = sum(values[-100:]) / min(100, len(values))
                if 'recon' in name or 'motion' in name:
                    summary['best_losses'][name] = min(values)
                else:
                    summary['best_losses'][name] = sum(values[-100:]) / min(100, len(values))

        with open(self.output_dir / 'training_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_module(self, model: nn.Module) -> nn.Module:
        """Unwrap DataParallel to access the underlying module."""
        return model.module if hasattr(model, 'module') else model

    def get_condition_frames(self, n_total: int) -> int:
        """
        Sample the number of condition frames for this training step.

        If random_condition=True, k is drawn uniformly from condition_frames_range.
        Otherwise n_condition_frames is used (clamped to leave at least one
        future frame to predict).
        """
        if self.random_condition:
            lo, hi = self.condition_frames_range
            hi = min(hi, n_total - 1)
            lo = min(lo, hi)
            return random.randint(lo, hi)
        return min(self.n_condition_frames, n_total - 1)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def D_step(
        self,
        real_video: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        Single discriminator update step.

        Args:
            real_video: [B, T, C, H, W] — real video batch.
            labels:     [B, c_dim]       — class one-hot labels.

        Returns:
            (D_loss_real_2d, D_loss_fake_2d, D_loss_real_3d, D_loss_fake_3d)
        """
        batch_size, n_frames = real_video.shape[:2]
        generator = self._get_module(self.generator)

        k = self.get_condition_frames(n_frames)
        condition_frames = real_video[:, :k]

        # Generate fake video without tracking gradients.
        generator.eval()
        with torch.no_grad():
            pred_video, _, _ = generator.predict_from_frames(
                early_frames=condition_frames,
                c=labels,
                total_frames=n_frames,
                return_z_seq=True,
            )
        generator.train()

        # --- 3D VideoDiscriminator ---
        real_input_3d = flip_video(prepare_video_disc_input(real_video))
        fake_input_3d = flip_video(prepare_video_disc_input(pred_video.detach()))

        D_real_3d = self.video_disc(real_input_3d)
        D_fake_3d = self.video_disc(fake_input_3d)

        D_loss_real_3d = self.criterion_gan(D_real_3d, D_fake_3d, True)
        D_loss_fake_3d = self.criterion_gan(D_fake_3d, D_real_3d, False)
        D_loss_3d = (D_loss_real_3d + D_loss_fake_3d) * 0.5

        loss_GP_3d = compute_gradient_penalty_3d(
            real_input_3d, fake_input_3d, self.video_disc, self.num_D
        )
        D_loss_3d = D_loss_3d + loss_GP_3d

        self._get_module(self.video_disc).optim.zero_grad()
        D_loss_3d.backward()
        self._get_module(self.video_disc).optim.step()

        # --- 2D ImageDiscriminator (only on predicted frames, not condition frames) ---
        frame_id = random.randint(k, n_frames - 1)

        real_input_2d = prepare_image_disc_input(real_video, frame_id)
        fake_input_2d = prepare_image_disc_input(pred_video.detach(), frame_id)

        D_real_2d = self.image_disc(real_input_2d)
        D_fake_2d = self.image_disc(fake_input_2d)

        D_loss_real_2d = self.criterion_gan(D_real_2d, D_fake_2d, True)
        D_loss_fake_2d = self.criterion_gan(D_fake_2d, D_real_2d, False)
        D_loss_2d = (D_loss_real_2d + D_loss_fake_2d) * 0.5

        self._get_module(self.image_disc).optim.zero_grad()
        D_loss_2d.backward()
        self._get_module(self.image_disc).optim.step()

        return (
            D_loss_real_2d.item(),
            D_loss_fake_2d.item(),
            D_loss_real_3d.item(),
            D_loss_fake_3d.item(),
        )

    def G_step(
        self,
        real_video: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, ...]:
        """
        Single generator (Encoder + MotionRNN) update step.

        Computes the full composite loss and updates the generator via
        self.opt_generator.

        Args:
            real_video: [B, T, C, H, W] — real video batch.
            labels:     [B, c_dim]       — class one-hot labels.

        Returns:
            9-tuple of scalar losses:
            (recon, adv_2d, adv_3d, motion, temporal, mutual, feat, kl, contrastive)
        """
        batch_size, n_frames = real_video.shape[:2]
        generator = self._get_module(self.generator)

        k = self.get_condition_frames(n_frames)
        condition_frames = real_video[:, :k]

        # Forward: predict full video and retrieve auxiliary outputs for losses.
        result = generator.predict_from_frames(
            early_frames=condition_frames,
            c=labels,
            total_frames=n_frames,
            return_z_seq=True,
            return_noise=True,
            return_kl_params=self.use_kl_loss,
        )

        if self.use_kl_loss:
            pred_video, z_seq, deltas, noise_in, noise_rec, mu, log_var = result
        else:
            pred_video, z_seq, deltas, noise_in, noise_rec = result
            mu, log_var = None, None

        # --- Reconstruction loss ---
        if self.use_progressive_recon:
            # Decaying weights for later frames to mitigate error accumulation.
            recon_loss = compute_progressive_recon_loss(
                pred_video, real_video, k, self.progressive_decay
            )
        else:
            # L1 only on the predicted (future) frames.
            recon_loss = F.l1_loss(pred_video[:, k:], real_video[:, k:])

        # --- Temporal direction loss ---
        temporal_loss = compute_temporal_direction_loss(pred_video, real_video)

        # --- Mutual information loss (noise injection) ---
        mutual_loss = compute_mutual_information_loss(noise_in, noise_rec)
        if not isinstance(mutual_loss, torch.Tensor):
            mutual_loss = torch.tensor(mutual_loss, device=self.device)

        # --- 2D adversarial loss ---
        frame_id = random.randint(k, n_frames - 1)
        fake_input_2d = prepare_image_disc_input(pred_video, frame_id)
        real_input_2d = prepare_image_disc_input(real_video, frame_id)

        D_fake_2d = self.image_disc(fake_input_2d)
        D_real_2d = self.image_disc(real_input_2d)

        adv_loss_2d = (
            self.criterion_gan(D_fake_2d, D_real_2d, True) +
            self.criterion_gan(D_real_2d, D_fake_2d, False)
        ) * 0.5

        # --- 3D adversarial loss + feature matching ---
        fake_input_3d = prepare_video_disc_input(pred_video)
        real_input_3d = prepare_video_disc_input(real_video)

        # No flip here so that real and fake features are comparable.
        D_fake_3d = self.video_disc(fake_input_3d)
        D_real_3d = self.video_disc(real_input_3d)

        adv_loss_3d = (
            self.criterion_gan(D_fake_3d, D_real_3d, True) +
            self.criterion_gan(D_real_3d, D_fake_3d, False)
        ) * 0.5

        if self.use_feature_matching:
            feat_loss = compute_feature_matching_loss(D_real_3d, D_fake_3d, self.num_D)
        else:
            feat_loss = torch.tensor(0.0, device=self.device)

        # --- Motion smoothness loss ---
        if deltas is not None and deltas.shape[1] > 0:
            motion_loss = generator.compute_motion_loss(deltas)
        else:
            motion_loss = torch.tensor(0.0, device=self.device)

        # --- KL divergence loss (VAE regularisation) ---
        kl_loss = (
            compute_kl_loss(mu, log_var)
            if (self.use_kl_loss and mu is not None)
            else torch.tensor(0.0, device=self.device)
        )

        # --- Contrastive identity-preservation loss (MoCoGAN-HD F network) ---
        if self.use_contrastive and self.contrastive_encoder is not None:
            # Anchor: first real frame (stable identity reference).
            real_first = real_video[:, 0]
            # Positive: a randomly chosen predicted frame (should share identity).
            random_t = torch.randint(1, n_frames, (1,)).item()
            pred_frame = pred_video[:, random_t]

            # Optional horizontal flip augmentation.
            if random.random() > 0.5:
                real_first = torch.flip(real_first, [-1])
                pred_frame = torch.flip(pred_frame, [-1])

            anchor_feats = self.contrastive_encoder(real_first)
            positive_feats = self.contrastive_encoder(pred_frame)

            memory = self.memory_bank.get()
            if memory is not None and memory.sum() != 0:
                contrastive_loss = compute_contrastive_loss_with_memory(
                    anchor_feats, positive_feats, memory
                )
            else:
                # Fall back to in-batch negatives early in training.
                contrastive_loss = compute_contrastive_loss(anchor_feats, positive_feats)

            # Store real frame features as negatives for future steps.
            self.memory_bank.update(anchor_feats)
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)

        # --- Total weighted loss ---
        total_loss = (
            self.lambda_recon       * recon_loss      +
            self.lambda_adv         * (adv_loss_2d + adv_loss_3d) +
            self.lambda_motion      * motion_loss     +
            self.lambda_temporal    * temporal_loss   +
            self.lambda_mutual      * mutual_loss     +
            self.lambda_feat        * feat_loss       +
            self.lambda_kl          * kl_loss         +
            self.lambda_contrastive * contrastive_loss
        )

        self.opt_generator.zero_grad()
        total_loss.backward()
        self.opt_generator.step()

        def _item(t):
            return t.item() if isinstance(t, torch.Tensor) else float(t)

        return (
            _item(recon_loss),
            _item(adv_loss_2d),
            _item(adv_loss_3d),
            _item(motion_loss),
            _item(temporal_loss),
            _item(mutual_loss),
            _item(feat_loss),
            _item(kl_loss),
            _item(contrastive_loss),
        )

    def GD_step(
        self,
        real_video: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[list, list]:
        """
        Full G+D training step (n_G_steps generator updates, then one D update).

        Args:
            real_video: [B, T, C, H, W]
            labels:     [B, c_dim]

        Returns:
            (loss_all, loss_names): paired scalar lists.
        """
        for _ in range(self.n_G_steps):
            (recon_loss, adv_loss_2d, adv_loss_3d, motion_loss,
             temporal_loss, mutual_loss, feat_loss, kl_loss,
             contrastive_loss) = self.G_step(real_video, labels)

        (D_loss_real_2d, D_loss_fake_2d,
         D_loss_real_3d, D_loss_fake_3d) = self.D_step(real_video, labels)

        loss_names = [
            'D_real_2d', 'D_fake_2d', 'D_real_3d', 'D_fake_3d',
            'recon', 'adv_2d', 'adv_3d', 'motion', 'temporal',
            'mutual', 'feat', 'kl', 'contrastive',
        ]
        loss_all = [
            D_loss_real_2d, D_loss_fake_2d, D_loss_real_3d, D_loss_fake_3d,
            recon_loss, adv_loss_2d, adv_loss_3d, motion_loss, temporal_loss,
            mutual_loss, feat_loss, kl_loss, contrastive_loss,
        ]

        return loss_all, loss_names

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_kimg: int = 2000,
        log_interval: int = 100,
        save_interval: int = 50,
        sample_interval: int = 25,
        n_sample_videos: int = 4,
    ):
        """
        Main conditional training loop.

        Args:
            total_kimg:      Training budget in kilo-images.
            log_interval:    Print and log losses every N steps.
            save_interval:   Save checkpoint every N kimg.
            sample_interval: Generate sample videos every N kimg.
            n_sample_videos: Number of videos per class to sample.
        """
        print("\nStarting V3Cell conditional training...")
        print(f"  Output dir:        {self.output_dir}")
        print(f"  Total kimg:        {total_kimg}")
        print(f"  Condition frames:  {self.condition_frames_range}")

        self.generator.train()

        data_iter = iter(self.dataloader)
        batch_size = self.dataloader.batch_size

        pbar = tqdm(total=total_kimg, desc="Training", unit="kimg")
        pbar.update(int(self.cur_kimg))

        while self.cur_kimg < total_kimg:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            real_video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)

            loss_all, loss_names = self.GD_step(real_video, labels)

            self.cur_step += 1
            self.cur_kimg += batch_size * self.n_frames / 1000.0

            for name, loss in zip(loss_names, loss_all):
                self.stats[name].append(loss)

            if self.cur_step % log_interval == 0:
                avg_losses = {}
                log_str = f"[Step {self.cur_step}] [kimg {self.cur_kimg:.1f}] "
                for name in loss_names:
                    if self.stats[name]:
                        avg = sum(self.stats[name][-log_interval:]) / min(log_interval, len(self.stats[name]))
                        avg_losses[name] = avg
                        log_str += f"{name}: {avg:.4f} "
                tqdm.write(log_str)
                self._log_to_csv(self.cur_step, self.cur_kimg, avg_losses)
                self._log_to_tensorboard(self.cur_step, avg_losses)

            # Checkpoint saving (once per save_interval kimg).
            if self.cur_kimg >= (int(self.cur_kimg / save_interval) + 1) * save_interval \
                    - batch_size * self.n_frames / 1000.0:
                self.save_checkpoint(f"checkpoint_{int(self.cur_kimg):06d}.pt")

            # Video sampling.
            if self.cur_kimg >= (int(self.cur_kimg / sample_interval) + 1) * sample_interval \
                    - batch_size * self.n_frames / 1000.0:
                self.sample_videos(n_sample_videos, sample_by_class=True)

            pbar.update(batch_size * self.n_frames / 1000.0)

        pbar.close()

        if self.writer and self.writer is not False:
            self.writer.close()

        self._save_final_stats()

        print("Training completed!")
        print(f"  Logs:      {self.log_file}")
        print(f"  TensorBoard: {self.tensorboard_dir}")
        print(f"  Config:    {self.config_file}")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str):
        """
        Save trainable model weights and training state.

        Only the trainable components (Encoder + MotionRNN + ContrastiveEncoder)
        are saved; the large frozen R3GAN weights are excluded.
        """
        path = self.output_dir / filename
        generator = self._get_module(self.generator)

        checkpoint = {
            'generator': generator.get_trainable_state_dict(),
            'image_disc': self._get_module(self.image_disc).state_dict(),
            'video_disc': self._get_module(self.video_disc).state_dict(),
            'opt_generator': self.opt_generator.state_dict(),
            'cur_step': self.cur_step,
            'cur_kimg': self.cur_kimg,
        }

        if self.contrastive_encoder is not None:
            checkpoint['contrastive_encoder'] = self.contrastive_encoder.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Restore model weights and training state from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        generator = self._get_module(self.generator)

        generator.load_trainable_state_dict(ckpt['generator'])
        self._get_module(self.image_disc).load_state_dict(ckpt['image_disc'])
        self._get_module(self.video_disc).load_state_dict(ckpt['video_disc'])

        if 'opt_generator' in ckpt:
            self.opt_generator.load_state_dict(ckpt['opt_generator'])
        if 'contrastive_encoder' in ckpt and self.contrastive_encoder is not None:
            self.contrastive_encoder.load_state_dict(ckpt['contrastive_encoder'])

        self.cur_step = ckpt.get('cur_step', 0)
        self.cur_kimg = ckpt.get('cur_kimg', 0.0)
        print(f"Loaded checkpoint: {path} (step={self.cur_step}, kimg={self.cur_kimg:.1f})")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_videos(self, n_videos: int = 4, sample_by_class: bool = True):
        """
        Generate and save sample videos (real vs predicted) for visual inspection.

        Args:
            n_videos:        Total number of videos to save.
            sample_by_class: If True, collect equal numbers of videos per class.
        """
        from utils.misc import save_video_as_gif

        self.generator.eval()
        sample_dir = self.output_dir / 'samples'
        sample_dir.mkdir(exist_ok=True)

        if sample_by_class:
            class_names = {0: 'lumen', 1: 'lumen_collapse', 2: 'non_lumen'}
            samples_per_class = max(1, n_videos // len(class_names))
            class_samples = {cid: [] for cid in class_names}

            for batch in self.dataloader:
                videos, labels = batch['video'], batch['label']
                for i in range(videos.shape[0]):
                    label_idx = torch.argmax(labels[i]).item()
                    if len(class_samples[label_idx]) < samples_per_class:
                        class_samples[label_idx].append((videos[i], labels[i]))
                if all(len(v) >= samples_per_class for v in class_samples.values()):
                    break

            sample_idx = 0
            for class_id in sorted(class_samples.keys()):
                class_name = class_names[class_id]
                generator = self._get_module(self.generator)
                for video, label in class_samples[class_id][:samples_per_class]:
                    real_v = video.unsqueeze(0).to(self.device)
                    label_t = label.unsqueeze(0).to(self.device)
                    pred_v = generator.predict_from_frames(
                        early_frames=real_v[:, :self.n_condition_frames],
                        c=label_t,
                        total_frames=self.n_frames,
                    )
                    save_video_as_gif(
                        real_v[0],
                        sample_dir / f"step{self.cur_step:06d}_{class_name}_real_{sample_idx}.gif",
                    )
                    save_video_as_gif(
                        pred_v[0],
                        sample_dir / f"step{self.cur_step:06d}_{class_name}_pred_{sample_idx}.gif",
                    )
                    sample_idx += 1

            print(f"Saved {sample_idx} sample videos (balanced by class) to {sample_dir}")

        else:
            batch = next(iter(self.dataloader))
            n = min(n_videos, batch['video'].shape[0])
            real_video = batch['video'][:n].to(self.device)
            labels = batch['label'][:n].to(self.device)

            generator = self._get_module(self.generator)
            pred_video = generator.predict_from_frames(
                early_frames=real_video[:, :self.n_condition_frames],
                c=labels,
                total_frames=self.n_frames,
            )

            for i in range(n):
                save_video_as_gif(
                    real_video[i],
                    sample_dir / f"step{self.cur_step:06d}_real_{i}.gif",
                )
                save_video_as_gif(
                    pred_video[i],
                    sample_dir / f"step{self.cur_step:06d}_pred_{i}.gif",
                )
            print(f"Saved {n} sample videos to {sample_dir}")

        self.generator.train()
