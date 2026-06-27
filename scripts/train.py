#!/usr/bin/env python3
"""
V3Cell unconditional training script (MoCoGAN-HD specification).

Trains MotionRNN from scratch to generate videos from sampled noise z,
without conditioning on observed early frames.  For the conditional
variant used in the paper see scripts/train_conditional.py.

Usage:
    python scripts/train.py \\
        --config configs/default.yaml \\
        --r3gan_ckpt checkpoints/training-runs/00002-frames-256-gpus3-batch96/network-snapshot-000000580.pkl \\
        --data_dir datasets/vedio_8x \\
        --output_dir checkpoints/outputs/v3cell
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from models import V3CellGenerator, ImageDiscriminator, VideoDiscriminator
from data import VideoDataset
from training import V3CellTrainer
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3Cell unconditional generator (MoCoGAN-HD style)')

    # Required arguments
    parser.add_argument('--r3gan_ckpt', type=str,
                        default='checkpoints/training-runs/00002-frames-256-gpus3-batch96/network-snapshot-000000580.pkl',
                        help='Path to R3GAN checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/vedio_8x',
                        help='Path to video data directory')

    # Optional arguments
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/outputs/v3cell',
                        help='Output directory')
    # R3GAN is bundled with the project; an external path is rarely needed
    parser.add_argument('--r3gan_path', type=str, default=None,
                        help='Path to external R3GAN source (optional, uses built-in by default)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Config override arguments
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('--total_kimg', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ids to use, e.g., "0,1,2,3,4,5,6,7" for 8 GPUs')

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure multi-GPU
    gpu_ids = [int(x) for x in args.gpu.split(',')]
    num_gpus = len(gpu_ids)

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs: {gpu_ids}")
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    config_path = project_root / args.config
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {
            'model': {
                'z_dim': 64,
                'c_dim': 4,
                'img_resolution': 64,
                'img_channels': 3,
                'motion_rnn': {
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'dropout': 0.1,
                    'residual_weight': 0.2,
                },
                'discriminator': {
                    'image': {'ndf': 64, 'n_layers': 3, 'norm_type': 'instance'},
                    'video': {'ndf': 64, 'n_layers': 3, 'norm_type': 'instance', 'num_D': 2},
                },
            },
            'training': {
                'batch_size': 8,
                'n_frames': 16,
                'total_kimg': 1000,
                'G_step': 1,
                'optimizer': {
                    'lr': 0.0002,
                    'beta1': 0.5,
                    'beta2': 0.999,
                },
                'lambda_gp': 10.0,
                'log_interval': 100,
                'save_interval': 10,
                'sample_interval': 5,
                'n_sample_videos': 4,
            },
            'data': {
                'frame_pattern': 'frame_*.png',
                'min_frames': 16,
                'augmentation': {
                    'horizontal_flip': True,
                    'vertical_flip': False,
                },
            },
            'device': {
                'num_workers': 4,
                'pin_memory': True,
            },
        }

    # Override config with command-line arguments
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.n_frames is not None:
        config['training']['n_frames'] = args.n_frames
    if args.total_kimg is not None:
        config['training']['total_kimg'] = args.total_kimg

    # Extract config sections
    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config['data']
    device_cfg = config.get('device', {})

    print("\n" + "="*60)
    print("V3Cell Unconditional Training (MoCoGAN-HD Style)")
    print("="*60)
    print(f"  z_dim: {model_cfg['z_dim']}")
    print(f"  c_dim: {model_cfg['c_dim']}")
    print(f"  batch_size: {train_cfg['batch_size']}")
    print(f"  n_frames: {train_cfg['n_frames']}")
    print(f"  total_kimg: {train_cfg['total_kimg']}")
    print(f"  G_step: {train_cfg.get('G_step', 1)}")
    print(f"  optimizer: Adam(lr={train_cfg['optimizer']['lr']}, beta1={train_cfg['optimizer']['beta1']})")
    print("="*60 + "\n")

    # Create dataset
    print(f"Loading data from: {args.data_dir}")
    dataset = VideoDataset(
        video_dir=args.data_dir,
        n_frames=train_cfg['n_frames'],
        img_resolution=model_cfg['img_resolution'],
        c_dim=model_cfg['c_dim'],
        frame_pattern=data_cfg.get('frame_pattern', 'frame_*.png'),
        min_frames=data_cfg.get('min_frames', 16),
        max_frames=data_cfg.get('max_frames', None),
        random_start=data_cfg.get('random_start', False),  # Start from frame 0 to learn the full developmental process
        horizontal_flip=data_cfg.get('augmentation', {}).get('horizontal_flip', True),
        vertical_flip=data_cfg.get('augmentation', {}).get('vertical_flip', False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=device_cfg.get('num_workers', 4),
        pin_memory=device_cfg.get('pin_memory', True),
        drop_last=True,
    )

    print(f"  Dataset size: {len(dataset)} videos")
    print(f"  Batches per epoch: {len(dataloader)}")

    # Create models
    print("\nCreating models...")

    # Optimizer parameters
    opt_cfg = train_cfg['optimizer']
    lr = opt_cfg['lr']
    beta1 = opt_cfg['beta1']
    beta2 = opt_cfg['beta2']

    # V3CellGenerator: frozen R3GAN + trainable MotionRNN
    generator = V3CellGenerator(
        r3gan_checkpoint=args.r3gan_ckpt,
        z_dim=model_cfg['z_dim'],
        c_dim=model_cfg['c_dim'],
        motion_hidden_dim=model_cfg['motion_rnn']['hidden_dim'],
        motion_num_layers=model_cfg['motion_rnn']['num_layers'],
        motion_dropout=model_cfg['motion_rnn']['dropout'],
        residual_weight=model_cfg['motion_rnn']['residual_weight'],
        freeze_r3gan=True,
        r3gan_path=args.r3gan_path,
    )

    # ImageDiscriminator (MoCoGAN-HD frame-pair discriminator)
    img_disc_cfg = model_cfg['discriminator']['image']
    image_disc = ImageDiscriminator(
        img_resolution=model_cfg['img_resolution'],
        img_channels=model_cfg.get('img_channels', 3),
        ndf=img_disc_cfg.get('ndf', 64),
        n_layers=img_disc_cfg.get('n_layers', 3),
        norm_type=img_disc_cfg.get('norm_type', 'instance'),
        lr=lr,
        beta1=beta1,
        beta2=beta2,
    )

    # VideoDiscriminator (MoCoGAN-HD multi-scale 3D discriminator)
    vid_disc_cfg = model_cfg['discriminator']['video']
    video_disc = VideoDiscriminator(
        img_channels=model_cfg.get('img_channels', 3),
        n_frames=train_cfg['n_frames'],
        ndf=vid_disc_cfg.get('ndf', 64),
        n_layers=vid_disc_cfg.get('n_layers', 3),
        norm_type=vid_disc_cfg.get('norm_type', 'instance'),
        num_D=vid_disc_cfg.get('num_D', 2),
        lr=lr,
        beta1=beta1,
        beta2=beta2,
    )

    from utils.misc import count_parameters
    print(f"  MotionRNN parameters:         {count_parameters(generator.motion_rnn):,}")
    print(f"  ImageDiscriminator parameters:{count_parameters(image_disc):,}")
    print(f"  VideoDiscriminator parameters:{count_parameters(video_disc):,}")

    generator = generator.to(device)
    image_disc = image_disc.to(device)
    video_disc = video_disc.to(device)

    if num_gpus > 1:
        print(f"\nDataParallel on GPUs: {gpu_ids}")
        generator = torch.nn.DataParallel(generator, device_ids=gpu_ids)
        image_disc = torch.nn.DataParallel(image_disc, device_ids=gpu_ids)
        video_disc = torch.nn.DataParallel(video_disc, device_ids=gpu_ids)

    # Optimiser covers only trainable MotionRNN parameters (R3GAN is frozen).
    motion_params = (
        generator.module.get_motion_params() if num_gpus > 1
        else generator.get_motion_params()
    )
    opt_motion = torch.optim.Adam(motion_params, lr=lr, betas=(beta1, beta2))

    trainer = V3CellTrainer(
        generator=generator,
        image_discriminator=image_disc,
        video_discriminator=video_disc,
        dataloader=dataloader,
        device=device,
        n_frames=train_cfg['n_frames'],
        G_step=train_cfg.get('G_step', 1),
        num_D=vid_disc_cfg.get('num_D', 2),
        output_dir=args.output_dir,
    )

    trainer.opt_motion = opt_motion

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\nStarting training...")
    try:
        trainer.train(
            total_kimg=train_cfg['total_kimg'],
            log_interval=train_cfg['log_interval'],
            save_interval=train_cfg['save_interval'],
            sample_interval=train_cfg['sample_interval'],
            n_sample_videos=train_cfg['n_sample_videos'],
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint('checkpoint_interrupted.pt')
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.save_checkpoint('checkpoint_error.pt')
        raise

    print("Done!")


if __name__ == '__main__':
    main()
