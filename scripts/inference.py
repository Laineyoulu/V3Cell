#!/usr/bin/env python3
"""
V3Cell conditional inference script.

Given the first n frames of an organoid video, predict the full developmental
trajectory using a trained V3CellGenerator checkpoint.

Usage:
    python scripts/inference.py \\
        --checkpoint checkpoints/outputs/v3cell/checkpoint_002000.pt \\
        --r3gan_ckpt checkpoints/training-runs/00002-frames-256-gpus3-batch96/network-snapshot-000000580.pkl \\
        --test_dir datasets/vedio_8x \\
        --output_dir outputs/inference_results \\
        --n_condition_frames 6 \\
        --gpu 0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from tqdm import tqdm

from models import V3CellGenerator
from utils.misc import save_video_as_gif


def parse_args():
    parser = argparse.ArgumentParser(description='V3Cell Conditional Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--r3gan_ckpt', type=str, required=True,
                        help='Path to R3GAN checkpoint')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/inference_results',
                        help='Output directory')
    parser.add_argument('--n_condition_frames', type=int, default=6,
                        help='Number of condition frames')
    parser.add_argument('--total_frames', type=int, default=25,
                        help='Total frames to generate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id')
    parser.add_argument('--img_resolution', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--save_individual_frames', action='store_true',
                        help='Save individual frames as PNG')
    parser.add_argument('--classes', nargs='+',
                        default=['non_lumen', 'lumen', 'lumen_collapse'],
                        choices=['lumen', 'lumen_collapse', 'non_lumen'],
                        help='Class folders to process')

    # Architecture flags — must match the settings used to train --checkpoint
    # (see scripts/train_conditional.py, which defaults to content-motion
    # separation with content_dim=32).
    parser.add_argument('--use_content_motion', action='store_true', default=True,
                        help='Build the generator with MoCoGAN-style content/motion latent separation')
    parser.add_argument('--no_content_motion', action='store_true',
                        help='Disable content-motion separation (use a single shared z_dim)')
    parser.add_argument('--content_dim', type=int, default=32,
                        help='Content latent dimension (only used if content-motion separation is enabled)')

    return parser.parse_args()


def load_model(checkpoint_path: str, r3gan_ckpt: str, device: torch.device,
               use_content_motion: bool = True, content_dim: int = 32) -> V3CellGenerator:
    """Load a trained V3CellGenerator from a checkpoint file."""

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Default model configuration (used if not stored in the checkpoint)
    z_dim = 64
    c_dim = 3
    img_resolution = 256
    motion_hidden_dim = 384
    motion_num_layers = 2
    motion_dropout = 0.1

    # Create model
    generator = V3CellGenerator(
        r3gan_checkpoint=r3gan_ckpt,
        z_dim=z_dim,
        c_dim=c_dim,
        img_resolution=img_resolution,
        motion_hidden_dim=motion_hidden_dim,
        motion_num_layers=motion_num_layers,
        motion_dropout=motion_dropout,
        residual_weight=0.2,
        use_content_motion=use_content_motion,
        content_dim=content_dim,
        use_encoder=True,
        freeze_r3gan=True,
    )

    generator = generator.to(device)

    # Load trainable weights (MotionRNN + Encoder only; R3GAN is frozen)
    generator.load_trainable_state_dict(ckpt['generator'])
    generator.eval()

    print(f"Model loaded successfully!")
    print(f"  Content-Motion separation: {use_content_motion}")
    print(f"  Content dim: {content_dim}, Motion dim: {z_dim - content_dim}")

    return generator


def load_video_frames(video_dir: Path, img_resolution: int = 256) -> Tuple[torch.Tensor, int]:
    """
    Load all frames from a directory, using the same preprocessing as training.

    Returns:
        frames: [T, C, H, W] tensor normalised to [-1, 1]
        total_frames: number of frames loaded
    """
    # Find all frame files
    frame_files = sorted(video_dir.glob("*.png"))

    if len(frame_files) == 0:
        raise ValueError(f"No frames found in {video_dir}")

    # Load and preprocess following the training procedure
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')

        # Resize to match training resolution
        if img.size != (img_resolution, img_resolution):
            img = img.resize(
                (img_resolution, img_resolution),
                Image.BILINEAR  # Same interpolation method as training
            )

        # Convert to numpy array
        frame = np.array(img)  # [H, W, C], uint8 in [0, 255]
        frames.append(frame)

    frames = np.stack(frames, axis=0)  # [T, H, W, C]

    # Convert to tensor (identical to training preprocessing)
    frames = frames.copy()  # Ensure contiguous memory
    frames = torch.from_numpy(frames).float()
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    frames = frames / 127.5 - 1.0  # Normalise to [-1, 1] (identical to training)

    return frames, len(frame_files)


def get_class_label(class_name: str) -> torch.Tensor:
    """Return the one-hot class label (must match the mapping used during training)."""
    # Must match the mapping in data/video_dataset.py
    class_map = {
        'lumen': 0,
        'lumen_collapse': 1,
        'non_lumen': 2,
    }

    if class_name not in class_map:
        raise ValueError(f"Unknown class: {class_name}")

    label = torch.zeros(3)
    label[class_map[class_name]] = 1.0

    return label


@torch.no_grad()
def inference_single_video(
    generator: V3CellGenerator,
    video_frames: torch.Tensor,
    class_label: torch.Tensor,
    n_condition_frames: int,
    total_frames: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run V3Cell conditional prediction on a single video.

    Args:
        generator: Trained V3CellGenerator.
        video_frames: [T, C, H, W] input video frames in [-1, 1].
        class_label: [c_dim] one-hot class label.
        n_condition_frames: Number of observed frames used as conditioning input.
        total_frames: Total number of frames to generate.
        device: Target device.

    Returns:
        condition_frames: [n_condition_frames, C, H, W] observed conditioning frames.
        predicted_video: [total_frames, C, H, W] predicted full trajectory.
    """
    # Use the first n frames as conditioning input
    condition_frames = video_frames[:n_condition_frames]  # [k, C, H, W]

    # Add batch dimension
    condition_frames_batch = condition_frames.unsqueeze(0).to(device)  # [1, k, C, H, W]
    class_label_batch = class_label.unsqueeze(0).to(device)           # [1, c_dim]

    # Generate video
    predicted_video = generator.predict_from_frames(
        early_frames=condition_frames_batch,
        c=class_label_batch,
        total_frames=total_frames,
    )  # [1, T, C, H, W]

    # Remove batch dimension
    predicted_video = predicted_video[0]  # [T, C, H, W]

    return condition_frames, predicted_video


def save_results(
    condition_frames: torch.Tensor,
    real_frames: torch.Tensor,
    predicted_frames: torch.Tensor,
    output_dir: Path,
    video_name: str,
    save_individual: bool = False,
):
    """
    Save inference results as GIFs (and optionally individual PNG frames).

    Args:
        condition_frames: [k, C, H, W] conditioning frames in [-1, 1].
        real_frames: [T, C, H, W] ground-truth frames in [-1, 1].
        predicted_frames: [T, C, H, W] predicted frames in [-1, 1].
        output_dir: Directory to write results into.
        video_name: Base name used for output files.
        save_individual: If True, also save each predicted frame as a PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as GIF (save_video_as_gif handles denormalisation automatically)
    save_video_as_gif(
        condition_frames,
        output_dir / f"{video_name}_condition.gif"
    )

    save_video_as_gif(
        real_frames,
        output_dir / f"{video_name}_real.gif"
    )

    save_video_as_gif(
        predicted_frames,
        output_dir / f"{video_name}_predicted.gif"
    )

    # Save individual frames (optional)
    if save_individual:
        frames_dir = output_dir / f"{video_name}_frames"
        frames_dir.mkdir(exist_ok=True)

        # Denormalise: [-1, 1] → [0, 255]
        frames_denorm = (predicted_frames + 1.0) / 2.0  # → [0, 1]
        frames_denorm = (frames_denorm * 255).clamp(0, 255)

        for i, frame in enumerate(frames_denorm):
            frame_np = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            img = Image.fromarray(frame_np)
            img.save(frames_dir / f"frame_{i:03d}.png")


def main():
    args = parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    use_content_motion = args.use_content_motion and not args.no_content_motion
    generator = load_model(args.checkpoint, args.r3gan_ckpt, device,
                            use_content_motion=use_content_motion,
                            content_dim=args.content_dim)

    # Test data directory
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")

    print(f"\nTest data directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Condition frames: {args.n_condition_frames}")
    print(f"Total frames: {args.total_frames}")
    print()

    # Iterate over classes
    class_names = args.classes
    all_results = []

    for class_name in class_names:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        print(f"\nProcessing class: {class_name}")
        print("=" * 60)

        # Get class label
        class_label = get_class_label(class_name)

        # Iterate over all videos in this class
        video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])

        for video_dir in tqdm(video_dirs, desc=f"  {class_name}"):
            video_name = video_dir.name

            try:
                # Load video frames
                video_frames, total_frames = load_video_frames(video_dir, args.img_resolution)

                if total_frames < args.n_condition_frames:
                    print(f"  Warning: {video_name} has only {total_frames} frames, skipping...")
                    continue

                # Run inference
                condition_frames, predicted_video = inference_single_video(
                    generator=generator,
                    video_frames=video_frames,
                    class_label=class_label,
                    n_condition_frames=args.n_condition_frames,
                    total_frames=args.total_frames,
                    device=device,
                )

                # Save results
                class_output_dir = output_dir / class_name
                save_results(
                    condition_frames=condition_frames,
                    real_frames=video_frames[:args.total_frames],
                    predicted_frames=predicted_video,
                    output_dir=class_output_dir,
                    video_name=video_name,
                    save_individual=args.save_individual_frames,
                )

                all_results.append({
                    'class': class_name,
                    'video': video_name,
                    'total_frames': total_frames,
                    'success': True,
                })

            except Exception as e:
                print(f"  Error processing {video_name}: {e}")
                all_results.append({
                    'class': class_name,
                    'video': video_name,
                    'success': False,
                    'error': str(e),
                })

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)

    success_count = sum(1 for r in all_results if r['success'])
    total_count = len(all_results)

    print(f"Total videos: {total_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"\nResults saved to: {output_dir}")

    # Per-class breakdown
    for class_name in class_names:
        class_results = [r for r in all_results if r['class'] == class_name]
        if class_results:
            class_success = sum(1 for r in class_results if r['success'])
            print(f"  {class_name}: {class_success}/{len(class_results)}")


if __name__ == '__main__':
    main()
