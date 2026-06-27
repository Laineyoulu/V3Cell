"""
V3CellTrainer: Unconditional video generation trainer (MoCoGAN-HD specification).

Training pipeline:
  1. D_step: Update discriminators with Relativistic Average LSGAN + gradient penalty.
  2. G_step: Update MotionRNN with adversarial loss + motion-smoothness MI loss.

The R3GAN image generator is frozen throughout; only MotionRNN parameters are updated.
This is the unconditional variant — videos are generated from a sampled noise z
without conditioning on observed early frames.  For the conditional variant used in
the V3Cell paper see training/conditional_trainer.py.
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
from training.losses import RelativisticAverageLSGAN, compute_gradient_penalty_3d


def flip_video(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly flip the video along the spatial (horizontal) axis.

    Used as data augmentation during discriminator training to reduce spatial bias.
    """
    if random.randint(0, 1) == 0:
        return torch.flip(x, [2])  # flip along the temporal/spatial axis
    return x


class V3CellTrainer:
    """
    Unconditional V3Cell trainer (MoCoGAN-HD specification).

    Training loop:
      - Loads real videos from the dataloader.
      - Generates fake videos using V3CellGenerator (sampled z → MotionRNN → R3GAN).
      - Trains 2D (ImageDiscriminator) and 3D (VideoDiscriminator) discriminators
        using Relativistic Average LSGAN + WGAN-GP gradient penalty.
      - Trains the MotionRNN with adversarial + motion-smoothness losses.
      - R3GAN parameters remain frozen throughout.

    Usage:
        trainer = V3CellTrainer(generator, image_disc, video_disc, dataloader, device)
        trainer.opt_motion = torch.optim.Adam(generator.get_motion_params(), ...)
        trainer.train(total_kimg=2000)
    """

    def __init__(
        self,
        generator: V3CellGenerator,
        image_discriminator: ImageDiscriminator,
        video_discriminator: VideoDiscriminator,
        dataloader: DataLoader,
        device: torch.device,
        n_frames: int = 8,
        G_step: int = 1,      # Number of G updates per D update
        num_D: int = 2,       # Number of multi-scale discriminator heads
        output_dir: Union[str, Path] = 'outputs',
    ):
        self.generator = generator.to(device)
        self.image_disc = image_discriminator.to(device)
        self.video_disc = video_discriminator.to(device)
        self.dataloader = dataloader
        self.device = device

        self.n_frames = n_frames
        self.n_G_steps = G_step
        self.num_D = num_D

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion_gan = RelativisticAverageLSGAN()

        self.cur_step = 0
        self.cur_kimg = 0.0
        self.stats = defaultdict(list)

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def D_step(
        self,
        real_video: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        Single discriminator update step.

        Args:
            real_video: [B, T, C, H, W] — real video batch.
            z:          [B, z_dim]       — initial noise (shared with G_step for comparison).

        Returns:
            (D_loss_real_2d, D_loss_fake_2d, D_loss_real_3d, D_loss_fake_3d)
        """
        batch_size = real_video.shape[0]

        # Generate fake videos (no gradient needed for discriminator update).
        with torch.no_grad():
            c = self.generator.sample_c(batch_size, self.device)
            fake_video, _, _ = self.generator(z, c, self.n_frames, return_z_seq=True)

        # --- 3D VideoDiscriminator update ---
        real_input_3d = prepare_video_disc_input(real_video)
        fake_input_3d = prepare_video_disc_input(fake_video)

        real_input_3d = flip_video(real_input_3d)
        fake_input_3d_flipped = flip_video(fake_input_3d)

        D_fake_3d = self.video_disc(fake_input_3d_flipped)
        D_real_3d = self.video_disc(real_input_3d)

        D_loss_real_3d = self.criterion_gan(D_real_3d, D_fake_3d, True)
        D_loss_fake_3d = self.criterion_gan(D_fake_3d, D_real_3d, False)
        D_loss_3d = (D_loss_real_3d + D_loss_fake_3d) * 0.5

        loss_GP_3d = compute_gradient_penalty_3d(
            real_input_3d, fake_input_3d.detach(),
            self.video_disc, self.num_D,
        )
        D_loss_3d = D_loss_3d + loss_GP_3d

        self.video_disc.optim.zero_grad()
        D_loss_3d.backward()
        self.video_disc.optim.step()

        # --- 2D ImageDiscriminator update ---
        frame_id = random.randint(1, self.n_frames - 1)

        real_input_2d = prepare_image_disc_input(real_video, frame_id)
        fake_input_2d = prepare_image_disc_input(fake_video.detach(), frame_id)

        D_real = self.image_disc(real_input_2d)
        D_fake = self.image_disc(fake_input_2d)

        D_loss_real = self.criterion_gan(D_real, D_fake, True)
        D_loss_fake = self.criterion_gan(D_fake, D_real, False)
        D_loss = (D_loss_real + D_loss_fake) * 0.5

        self.image_disc.optim.zero_grad()
        D_loss.backward()
        self.image_disc.optim.step()

        return (
            D_loss_real.item(),
            D_loss_fake.item(),
            D_loss_real_3d.item(),
            D_loss_fake_3d.item(),
        )

    def G_step(
        self,
        real_video: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Single generator (MotionRNN) update step.

        Noise z is re-sampled at each call, following MoCoGAN-HD convention.

        Args:
            real_video: [B, T, C, H, W] — real video (used for relativistic loss reference).

        Returns:
            (G_loss_2d, G_loss_3d, motion_loss)
        """
        batch_size = real_video.shape[0]

        z = self.generator.sample_z(batch_size, self.device)
        c = self.generator.sample_c(batch_size, self.device)
        fake_video, z_seq, deltas = self.generator(z, c, self.n_frames, return_z_seq=True)

        # Motion smoothness regularisation (acts as motion MI loss proxy).
        motion_loss = self.generator.compute_motion_loss(deltas)

        # --- 2D adversarial loss ---
        frame_id = random.randint(1, self.n_frames - 1)
        fake_input_2d = prepare_image_disc_input(fake_video, frame_id)
        real_input_2d = prepare_image_disc_input(real_video, frame_id)

        D_fake_2d = self.image_disc(fake_input_2d)
        D_real_2d = self.image_disc(real_input_2d)

        G_loss_2d = (
            self.criterion_gan(D_fake_2d, D_real_2d, True) +
            self.criterion_gan(D_real_2d, D_fake_2d, False)
        ) * 0.5

        # --- 3D adversarial loss ---
        real_input_3d = prepare_video_disc_input(real_video)
        fake_input_3d = prepare_video_disc_input(fake_video)

        D_real_3d = self.video_disc(flip_video(real_input_3d))
        D_fake_3d = self.video_disc(flip_video(fake_input_3d))

        G_loss_3d = (
            self.criterion_gan(D_fake_3d, D_real_3d, True) +
            self.criterion_gan(D_real_3d, D_fake_3d, False)
        ) * 0.5

        G_loss = G_loss_3d + G_loss_2d + motion_loss

        self.opt_motion.zero_grad()
        G_loss.backward()
        self.opt_motion.step()

        return G_loss_2d.item(), G_loss_3d.item(), motion_loss.item()

    def GD_step(
        self,
        real_video: torch.Tensor,
    ) -> Tuple[list, list]:
        """
        Full G+D training step (MoCoGAN-HD style).

        Performs self.G_step generator updates followed by one discriminator update.

        Args:
            real_video: [B, T, C, H, W]

        Returns:
            (loss_all, loss_names): paired lists of scalar loss values and names.
        """
        batch_size = real_video.shape[0]

        # Multiple G updates per D update (e.g. G_step=5 per MoCoGAN-HD).
        for _ in range(self.n_G_steps):
            G_loss_2d, G_loss_3d, motion_loss = self.G_step(real_video)

        z = self.generator.sample_z(batch_size, self.device)
        D_loss_real, D_loss_fake, D_loss_real_3d, D_loss_fake_3d = self.D_step(real_video, z)

        loss_names = ['D_real', 'D_fake', 'D_real_3d', 'D_fake_3d',
                      'G_2d', 'G_3d', 'motion']
        loss_all = [D_loss_real, D_loss_fake, D_loss_real_3d, D_loss_fake_3d,
                    G_loss_2d, G_loss_3d, motion_loss]

        return loss_all, loss_names

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        total_kimg: int = 1000,
        log_interval: int = 100,
        save_interval: int = 10,
        sample_interval: int = 5,
        n_sample_videos: int = 4,
    ):
        """
        Main training loop.

        Args:
            total_kimg:      Total training budget in kilo-images.
            log_interval:    Print losses every N steps.
            save_interval:   Save checkpoint every N kimg.
            sample_interval: Generate sample videos every N kimg.
            n_sample_videos: Number of sample videos per class to save.
        """
        batch_size = self.dataloader.batch_size
        images_per_step = batch_size * self.n_frames

        print("Starting V3Cell unconditional training (MoCoGAN-HD style)...")
        print(f"  Total kimg:           {total_kimg}")
        print(f"  Batch size:           {batch_size}")
        print(f"  Frames per video:     {self.n_frames}")
        print(f"  G steps per D step:   {self.n_G_steps}")

        data_iter = iter(self.dataloader)
        pbar = tqdm(total=total_kimg, desc='Training', unit='kimg')
        pbar.update(self.cur_kimg)

        last_save_kimg = 0.0
        last_sample_kimg = 0.0

        while self.cur_kimg < total_kimg:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            real_video = batch['video'].to(self.device)

            loss_all, loss_names = self.GD_step(real_video)

            for name, value in zip(loss_names, loss_all):
                self.stats[name].append(value)

            self.cur_step += 1
            self.cur_kimg += images_per_step / 1000.0

            if self.cur_step % log_interval == 0:
                log_str = f"[{self.cur_kimg:.1f}k] "
                log_str += " ".join(f"{n}: {v:.4f}" for n, v in zip(loss_names, loss_all))
                tqdm.write(log_str)

            if self.cur_kimg - last_save_kimg >= save_interval:
                self.save_checkpoint()
                last_save_kimg = self.cur_kimg

            if self.cur_kimg - last_sample_kimg >= sample_interval:
                self.sample_videos(n_sample_videos)
                last_sample_kimg = self.cur_kimg

            pbar.update(images_per_step / 1000.0)

        pbar.close()
        print("Training completed!")
        self.save_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: Optional[str] = None):
        """Save a full checkpoint (generator + discriminators + training state)."""
        if filename is None:
            filename = f"checkpoint_{self.cur_kimg:.0f}k.pt"

        checkpoint = {
            'generator': self.generator.state_dict(),
            'image_disc': self.image_disc.state_dict(),
            'video_disc': self.video_disc.state_dict(),
            'cur_step': self.cur_step,
            'cur_kimg': self.cur_kimg,
        }

        save_path = self.output_dir / 'checkpoints'
        save_path.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path / filename)
        print(f"Saved checkpoint: {save_path / filename}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Restore all model weights and training state from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator'])
        self.image_disc.load_state_dict(checkpoint['image_disc'])
        self.video_disc.load_state_dict(checkpoint['video_disc'])
        self.cur_step = checkpoint['cur_step']
        self.cur_kimg = checkpoint['cur_kimg']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Step: {self.cur_step}, kimg: {self.cur_kimg:.1f}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_videos(self, n_videos: int = 4):
        """
        Generate sample video frames for each class and save as image grids.

        Args:
            n_videos: Number of sample videos to generate per class.
        """
        self.generator.eval()

        sample_dir = self.output_dir / 'samples'
        sample_dir.mkdir(exist_ok=True)

        for class_idx in range(self.generator.c_dim):
            z = self.generator.sample_z(n_videos, self.device)
            c = self.generator.sample_c(n_videos, self.device, class_idx=class_idx)

            videos = self.generator(z, c, self.n_frames)
            videos = ((videos + 1) / 2).clamp(0, 1)

            for i, video in enumerate(videos):
                save_video_grid(
                    video,
                    sample_dir / f"kimg{self.cur_kimg:.0f}_class{class_idx}_sample{i}.png",
                )

        self.generator.train()
        print(f"Saved samples at kimg={self.cur_kimg:.0f}")


def save_video_grid(video: torch.Tensor, path: Union[str, Path], nrow: int = 8):
    """
    Save video frames as a horizontal image grid.

    Args:
        video: [T, C, H, W] — frame tensor in [0, 1].
        path:  Output image path.
        nrow:  Number of frames per row in the grid.
    """
    from torchvision.utils import make_grid
    from PIL import Image

    grid = make_grid(video, nrow=nrow, padding=2, normalize=False)
    grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    Image.fromarray(grid).save(path)
