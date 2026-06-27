#!/usr/bin/env python3
"""
V3Cell video generation (inference) script.

Loads a trained V3CellGenerator checkpoint and generates organoid development
videos unconditionally (noise → MotionRNN → R3GAN → frames).

Usage:
    python scripts/generate.py \\
        --checkpoint checkpoints/outputs/v3cell/checkpoint_002000.pt \\
        --r3gan_ckpt checkpoints/training-runs/00002-frames-256-gpus3-batch96/network-snapshot-000000580.pkl \\
        --n_videos 10 \\
        --n_frames 16 \\
        --output_dir outputs/generated
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm

from models import V3CellGenerator
from utils import load_config, set_seed, save_video_as_gif


def parse_args():
    parser = argparse.ArgumentParser(description='Generate videos with V3Cell')

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to V3Cell checkpoint (.pt)')
    parser.add_argument('--r3gan_ckpt', type=str, required=True,
                        help='Path to R3GAN checkpoint (.pkl)')

    # Optional paths
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default='outputs/generated',
                        help='Output directory for generated videos')
    parser.add_argument('--r3gan_path', type=str, default=None,
                        help='Path to external R3GAN source (usually not needed)')

    # Generation parameters
    parser.add_argument('--n_videos', type=int, default=10,
                        help='Total number of videos to generate')
    parser.add_argument('--n_frames', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('--class_idx', type=int, default=None,
                        help='Class index to generate (None = generate all classes)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')

    # Output format
    parser.add_argument('--format', type=str, default='gif',
                        choices=['gif', 'mp4', 'png'],
                        help='Output format: gif, mp4, or png (frame sequence)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for video output')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)

    # Architecture flags — must match the settings used to train --checkpoint
    # (see scripts/train_conditional.py, which defaults to content-motion
    # separation with content_dim=32; inference.py hardcodes the same values).
    parser.add_argument('--use_content_motion', action='store_true', default=True,
                        help='Build the generator with MoCoGAN-style content/motion latent separation')
    parser.add_argument('--no_content_motion', action='store_true',
                        help='Disable content-motion separation (use a single shared z_dim)')
    parser.add_argument('--content_dim', type=int, default=32,
                        help='Content latent dimension (only used if content-motion separation is enabled)')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(args.seed)

    # Load configuration.
    config_path = project_root / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path} — using built-in defaults.")
        config = {
            'model': {
                'z_dim': 64,
                'c_dim': 3,
                'img_resolution': 256,
                'motion_rnn': {
                    'hidden_dim': 384,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'residual_weight': 0.2,
                },
            },
        }

    model_cfg = config['model']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and load model.
    use_content_motion = args.use_content_motion and not args.no_content_motion
    print("Loading V3Cell model...")
    print(f"  Content-motion separation: {use_content_motion} (content_dim={args.content_dim})")
    generator = V3CellGenerator(
        r3gan_checkpoint=args.r3gan_ckpt,
        z_dim=model_cfg['z_dim'],
        c_dim=model_cfg['c_dim'],
        motion_hidden_dim=model_cfg['motion_rnn']['hidden_dim'],
        motion_num_layers=model_cfg['motion_rnn']['num_layers'],
        motion_dropout=model_cfg['motion_rnn']['dropout'],
        residual_weight=model_cfg['motion_rnn']['residual_weight'],
        use_content_motion=use_content_motion,
        content_dim=args.content_dim,
        freeze_r3gan=True,
        r3gan_path=args.r3gan_path,
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    generator.load_trainable_state_dict(checkpoint['generator'])
    generator = generator.to(device).eval()

    print(f"Loaded checkpoint: {args.checkpoint}")

    # Determine which classes to generate.
    class_indices = [args.class_idx] if args.class_idx is not None \
        else list(range(model_cfg['c_dim']))

    print(f"\nGenerating {args.n_videos} videos × {args.n_frames} frames "
          f"for classes {class_indices} ...")

    video_idx = 0
    with torch.no_grad():
        for class_idx in class_indices:
            # Distribute n_videos evenly across classes.
            n_class_videos = args.n_videos // len(class_indices)
            if class_idx < args.n_videos % len(class_indices):
                n_class_videos += 1

            print(f"\nClass {class_idx}: generating {n_class_videos} videos")

            for batch_start in tqdm(range(0, n_class_videos, args.batch_size)):
                batch_size = min(args.batch_size, n_class_videos - batch_start)

                z = generator.sample_z(batch_size, device)
                c = generator.sample_c(batch_size, device, class_idx=class_idx)

                # Generate and rescale from [-1, 1] to [0, 1].
                videos = generator(z, c, args.n_frames)
                videos = ((videos + 1) / 2).clamp(0, 1)

                for i, video in enumerate(videos):
                    if args.format == 'gif':
                        save_path = output_dir / f"video_{video_idx:04d}_class{class_idx}.gif"
                        save_video_as_gif(video, save_path, fps=args.fps)

                    elif args.format == 'mp4':
                        from utils.misc import save_video_as_mp4
                        save_path = output_dir / f"video_{video_idx:04d}_class{class_idx}.mp4"
                        save_video_as_mp4(video, save_path, fps=args.fps)

                    elif args.format == 'png':
                        # Save each frame as a separate PNG.
                        vid_dir = output_dir / f"video_{video_idx:04d}_class{class_idx}"
                        vid_dir.mkdir(exist_ok=True)
                        for t, frame in enumerate(video):
                            from torchvision.utils import save_image
                            save_image(frame, vid_dir / f"frame_{t:04d}.png")

                    video_idx += 1

    print(f"\nGenerated {video_idx} videos → {output_dir}")


if __name__ == '__main__':
    main()
