#!/usr/bin/env python3
"""
V3Cell conditional video generation training script.

Training objective:
  Given early organoid development frames, predict the complete developmental
  trajectory (full video).

Usage:
    python scripts/train_conditional.py \\
        --r3gan_ckpt checkpoints/training-runs/00002-frames-256-gpus3-batch96/network-snapshot-000000580.pkl \\
        --data_dir datasets/vedio_8x \\
        --output_dir checkpoints/outputs/v3cell \\
        --gpu 0,1,2,3
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is at the front of sys.path before any local imports.
project_root = Path(__file__).parent.parent.resolve()
sys.path = [p for p in sys.path if 'r3gan' not in p]
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))
sys.path.insert(0, str(project_root))

import torch

from models import V3CellGenerator, ImageDiscriminator, VideoDiscriminator
from data import VideoDataset
from training import ConditionalTrainer
from utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train V3Cell Conditional Video Generator')

    # Required arguments
    parser.add_argument('--r3gan_ckpt', type=str,
                        default='checkpoints/training-runs/00002-frames-256-gpus3-batch96/network-snapshot-000000580.pkl',
                        help='Path to the pre-trained R3GAN checkpoint (.pkl)')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/vedio_8x',
                        help='Root directory of the organoid video dataset')

    # Optional paths
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/outputs/v3cell',
                        help='Directory for checkpoints, samples and logs')
    parser.add_argument('--r3gan_path', type=str, default=None,
                        help='Path to external R3GAN source (usually not needed)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint to resume training from')

    # Training overrides
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--img_resolution', type=int, default=None,
                        help='Override spatial resolution (must match R3GAN training resolution)')
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('--n_condition_frames', type=int, default=10,
                        help='Number of observed early frames used as condition')
    parser.add_argument('--total_kimg', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0',
                        help='Comma-separated GPU ids, e.g. "0,1,2,3"')
    parser.add_argument('--G_step', type=int, default=None,
                        help='Generator updates per discriminator update (default: 5)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='MotionRNN LSTM hidden dimension (default: 384)')

    # Content-motion separation (MoCoGAN-style)
    parser.add_argument('--use_content_motion', action='store_true', default=True,
                        help='Use MoCoGAN content-motion separation (z = [z_C, z_M])')
    parser.add_argument('--no_content_motion', action='store_true',
                        help='Disable content-motion separation')
    parser.add_argument('--content_dim', type=int, default=32,
                        help='Content code dimension z_C (default: 32 for z_dim=64)')

    # Logging and saving intervals
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Print losses every N steps')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Save checkpoint every N kimg')
    parser.add_argument('--sample_interval', type=int, default=None,
                        help='Sample videos every N kimg')

    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=10.0,
                        help='Reconstruction loss weight (L1)')
    parser.add_argument('--lambda_adv', type=float, default=1.0,
                        help='Adversarial loss weight')
    parser.add_argument('--lambda_motion', type=float, default=0.1,
                        help='Motion smoothness loss weight')
    parser.add_argument('--lambda_temporal', type=float, default=1.0,
                        help='Temporal direction loss weight')
    parser.add_argument('--lambda_mutual', type=float, default=1.0,
                        help='Mutual information loss weight (prevents motion collapse)')
    parser.add_argument('--lambda_feat', type=float, default=10.0,
                        help='Feature matching loss weight (MoCoGAN-HD)')
    parser.add_argument('--lambda_kl', type=float, default=0.01,
                        help='KL divergence loss weight (VAE regularisation)')

    # Advanced options
    parser.add_argument('--use_progressive_recon', action='store_true', default=True,
                        help='Use decaying reconstruction weights for later frames')
    parser.add_argument('--no_progressive_recon', action='store_true',
                        help='Disable progressive reconstruction loss')
    parser.add_argument('--progressive_decay', type=float, default=0.95,
                        help='Per-frame decay factor for progressive reconstruction')
    parser.add_argument('--use_feature_matching', action='store_true', default=True,
                        help='Use discriminator feature matching loss (MoCoGAN-HD)')
    parser.add_argument('--no_feature_matching', action='store_true',
                        help='Disable feature matching loss')
    parser.add_argument('--use_kl_loss', action='store_true', default=True,
                        help='Use KL divergence loss to regularise the VAE encoder')
    parser.add_argument('--no_kl_loss', action='store_true',
                        help='Disable KL divergence loss')

    # Contrastive learning (MoCoGAN-HD F network)
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                        help='Use contrastive identity-preservation loss')
    parser.add_argument('--no_contrastive', action='store_true',
                        help='Disable contrastive learning loss')
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='Contrastive loss weight')
    parser.add_argument('--contrastive_feature_dim', type=int, default=128,
                        help='Feature dimension for the contrastive encoder F')
    parser.add_argument('--memory_bank_size', type=int, default=4096,
                        help='Number of entries in the contrastive memory bank')

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse GPU configuration.
    gpu_ids = [int(x) for x in args.gpu.split(',')]
    num_gpus = len(gpu_ids)
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

    if num_gpus > 1:
        print(f"Multi-GPU training on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    set_seed(args.seed)

    # Load configuration, falling back to built-in defaults if not found.
    config_path = project_root / args.config
    config = load_config(config_path) if config_path.exists() else get_default_config()
    if not config_path.exists():
        print(f"Config not found at {config_path} — using built-in defaults.")

    # Apply command-line overrides.
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.n_frames is not None:
        config['training']['n_frames'] = args.n_frames
    if args.total_kimg is not None:
        config['training']['total_kimg'] = args.total_kimg
    if args.G_step is not None:
        config['training']['G_step'] = args.G_step
    if args.hidden_dim is not None:
        config['model']['motion_rnn']['hidden_dim'] = args.hidden_dim
    if args.img_resolution is not None:
        config['model']['img_resolution'] = args.img_resolution
    if args.log_interval is not None:
        config['training']['log_interval'] = args.log_interval
    if args.save_interval is not None:
        config['training']['save_interval'] = args.save_interval
    if args.sample_interval is not None:
        config['training']['sample_interval'] = args.sample_interval

    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config['data']
    device_cfg = config.get('device', {})

    # Resolve boolean flags (--flag / --no_flag pattern).
    use_progressive_recon = args.use_progressive_recon and not args.no_progressive_recon
    use_content_motion = args.use_content_motion and not args.no_content_motion
    use_feature_matching = args.use_feature_matching and not args.no_feature_matching
    use_kl_loss = args.use_kl_loss and not args.no_kl_loss
    use_contrastive = args.use_contrastive and not args.no_contrastive

    frames_per_batch = train_cfg['batch_size'] * train_cfg['n_frames']

    print("\n" + "=" * 60)
    print("V3Cell Conditional Video Generation — Training")
    print("=" * 60)
    print(f"  Task:               Predict full development from early frames")
    print(f"  Condition frames:   {args.n_condition_frames}")
    print(f"  z_dim:              {model_cfg['z_dim']}")
    print(f"  c_dim:              {model_cfg['c_dim']}  (organoid classes)")
    print(f"  img_resolution:     {model_cfg['img_resolution']}")
    print(f"  batch_size:         {train_cfg['batch_size']}")
    print(f"  n_frames:           {train_cfg['n_frames']}")
    print(f"  frames_per_batch:   {frames_per_batch}")
    print(f"  total_kimg:         {train_cfg['total_kimg']}")
    print(f"  G_step:             {train_cfg.get('G_step', 1)}")
    print(f"  MotionRNN hidden:   {model_cfg['motion_rnn']['hidden_dim']}")
    print(f"  Content-motion:     {use_content_motion} (content_dim={args.content_dim})")
    print(f"  Loss weights:       recon={args.lambda_recon}, adv={args.lambda_adv}, "
          f"motion={args.lambda_motion}")
    print(f"                      temporal={args.lambda_temporal}, "
          f"mutual={args.lambda_mutual}")
    print(f"                      feat={args.lambda_feat}, kl={args.lambda_kl}, "
          f"contrastive={args.lambda_contrastive}")
    print(f"  Feature matching:   {use_feature_matching}")
    print(f"  KL loss:            {use_kl_loss}")
    print(f"  Contrastive:        {use_contrastive} (dim={args.contrastive_feature_dim})")
    print(f"  Progressive recon:  {use_progressive_recon} (decay={args.progressive_decay})")
    if frames_per_batch > 256:
        print(f"  WARNING: {frames_per_batch} frames/batch is large — "
              f"reduce batch_size if OOM occurs during G_step.")
    print("=" * 60 + "\n")

    # --- Dataset ---
    print(f"Loading dataset from: {args.data_dir}")
    dataset = VideoDataset(
        video_dir=args.data_dir,
        n_frames=train_cfg['n_frames'],
        img_resolution=model_cfg['img_resolution'],
        c_dim=model_cfg['c_dim'],
        frame_pattern=data_cfg.get('frame_pattern', '*_t*.png'),
        min_frames=data_cfg.get('min_frames', 16),
        max_frames=data_cfg.get('max_frames', None),
        random_start=data_cfg.get('random_start', False),
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

    print(f"  Dataset size:  {len(dataset)} videos")
    print(f"  Batches/epoch: {len(dataloader)}")

    # --- Models ---
    print("\nBuilding models...")

    opt_cfg = train_cfg['optimizer']
    lr, beta1, beta2 = opt_cfg['lr'], opt_cfg['beta1'], opt_cfg['beta2']

    generator = V3CellGenerator(
        r3gan_checkpoint=args.r3gan_ckpt,
        z_dim=model_cfg['z_dim'],
        c_dim=model_cfg['c_dim'],
        img_resolution=model_cfg['img_resolution'],
        motion_hidden_dim=model_cfg['motion_rnn']['hidden_dim'],
        motion_num_layers=model_cfg['motion_rnn']['num_layers'],
        motion_dropout=model_cfg['motion_rnn']['dropout'],
        residual_weight=model_cfg['motion_rnn']['residual_weight'],
        use_content_motion=use_content_motion,
        content_dim=args.content_dim,
        use_encoder=True,
        freeze_r3gan=True,
        r3gan_path=args.r3gan_path,
        device_ids=gpu_ids if num_gpus > 1 else None,
    )

    img_disc_cfg = model_cfg['discriminator']['image']
    image_disc = ImageDiscriminator(
        img_resolution=model_cfg['img_resolution'],
        img_channels=model_cfg.get('img_channels', 3),
        ndf=img_disc_cfg.get('ndf', 64),
        n_layers=img_disc_cfg.get('n_layers', 3),
        norm_type=img_disc_cfg.get('norm_type', 'instance'),
        lr=lr, beta1=beta1, beta2=beta2,
    )

    vid_disc_cfg = model_cfg['discriminator']['video']
    video_disc = VideoDiscriminator(
        img_channels=model_cfg.get('img_channels', 3),
        n_frames=train_cfg['n_frames'],
        ndf=vid_disc_cfg.get('ndf', 64),
        n_layers=vid_disc_cfg.get('n_layers', 3),
        norm_type=vid_disc_cfg.get('norm_type', 'instance'),
        num_D=vid_disc_cfg.get('num_D', 2),
        lr=lr, beta1=beta1, beta2=beta2,
    )

    from utils.misc import count_parameters
    print(f"  MotionRNN params:         {count_parameters(generator.motion_rnn):,}")
    print(f"  Encoder params:           {count_parameters(generator.encoder):,}")
    print(f"  ImageDiscriminator params:{count_parameters(image_disc):,}")
    print(f"  VideoDiscriminator params:{count_parameters(video_disc):,}")

    generator = generator.to(device)
    image_disc = image_disc.to(device)
    video_disc = video_disc.to(device)

    if num_gpus > 1:
        print(f"\nDataParallel on GPUs {gpu_ids}:")
        print(f"  R3GAN:        DataParallel (gradient flows to latents)")
        print(f"  Encoder/MRNN: primary GPU only")
        print(f"  Discriminators: DataParallel")
        image_disc = torch.nn.DataParallel(image_disc, device_ids=gpu_ids)
        video_disc = torch.nn.DataParallel(video_disc, device_ids=gpu_ids)

    # --- Trainer ---
    trainer = ConditionalTrainer(
        generator=generator,
        image_discriminator=image_disc,
        video_discriminator=video_disc,
        dataloader=dataloader,
        device=device,
        n_frames=train_cfg['n_frames'],
        n_condition_frames=args.n_condition_frames,
        condition_frames_range=(5, min(15, train_cfg['n_frames'] // 2)),
        random_condition=True,
        G_step=train_cfg.get('G_step', 5),
        num_D=vid_disc_cfg.get('num_D', 2),
        lambda_recon=args.lambda_recon,
        lambda_adv=args.lambda_adv,
        lambda_motion=args.lambda_motion,
        lambda_temporal=args.lambda_temporal,
        lambda_mutual=args.lambda_mutual,
        lambda_feat=args.lambda_feat,
        lambda_kl=args.lambda_kl,
        lambda_contrastive=args.lambda_contrastive,
        use_progressive_recon=use_progressive_recon,
        progressive_decay=args.progressive_decay,
        use_feature_matching=use_feature_matching,
        use_kl_loss=use_kl_loss,
        use_contrastive=use_contrastive,
        contrastive_feature_dim=args.contrastive_feature_dim,
        memory_bank_size=args.memory_bank_size,
        output_dir=args.output_dir,
    )

    # Build optimiser: covers Encoder + MotionRNN and optionally ContrastiveEncoder.
    trainable_params = list(generator.get_trainable_params())
    if use_contrastive and trainer.contrastive_encoder is not None:
        trainable_params += list(trainer.contrastive_encoder.parameters())
        print(f"  ContrastiveEncoder params:{count_parameters(trainer.contrastive_encoder):,}")

    trainer.opt_generator = torch.optim.Adam(
        trainable_params, lr=lr, betas=(beta1, beta2)
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # --- Train ---
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
        print("\nInterrupted by user — saving checkpoint.")
        trainer.save_checkpoint('checkpoint_interrupted.pt')
    except Exception as e:
        print(f"\nTraining failed: {e}")
        trainer.save_checkpoint('checkpoint_error.pt')
        raise

    print("Done!")


def get_default_config() -> dict:
    """
    Built-in default configuration aligned with MoCoGAN-HD settings.

    Used as a fallback when no YAML config file is found.
    """
    return {
        'model': {
            'z_dim': 64,           # R3GAN NoiseDimension
            'c_dim': 3,            # Number of organoid classes
            'img_resolution': 256,
            'img_channels': 3,
            'motion_rnn': {
                'hidden_dim': 384,  # MoCoGAN-HD: 384
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
            'n_frames': 16,        # MoCoGAN-HD: 16 frames per clip
            'total_kimg': 2000,
            'G_step': 5,           # MoCoGAN-HD: 5 G steps per D step
            'optimizer': {'lr': 0.0002, 'beta1': 0.5, 'beta2': 0.999},
            'log_interval': 100,
            'save_interval': 50,
            'sample_interval': 25,
            'n_sample_videos': 4,
        },
        'data': {
            'frame_pattern': '*_t*.png',
            'min_frames': 16,
            'max_frames': None,
            'random_start': False,
            'augmentation': {'horizontal_flip': True, 'vertical_flip': False},
        },
        'device': {'num_workers': 4, 'pin_memory': True},
    }


if __name__ == '__main__':
    main()
