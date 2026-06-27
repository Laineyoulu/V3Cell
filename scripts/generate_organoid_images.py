#!/usr/bin/env python3
"""
Organoid image generation script — generates images per organ type and class.

Supported organs and classes:
- colon: Budding_opaque, Budding_transparent, Non_budding_opaque, Non_budding_transparent
- lung: lumen, non_lumen
- stomach: Apoptotic, Hollow, Solid

Usage:
    # Generate for all organs, 10 runs, 2000 images per class per run
    python scripts/generate_organoid_images.py \\
        --runs 10 \\
        --images_per_class 2000 \\
        --outdir outputs/organoid_images

    # Generate for stomach only
    python scripts/generate_organoid_images.py \\
        --organ stomach \\
        --runs 5 \\
        --images_per_class 1000 \\
        --outdir outputs/organoid_images
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Dict, List

# Add r3gan to path (for `import legacy`, `from training import ...`),
# then the project root (for `import dnnlib`, which lives at the repo root).
project_root = Path(__file__).parent.parent.resolve()
r3gan_path = project_root / 'r3gan'
sys.path.insert(0, str(r3gan_path))
sys.path.insert(1, str(project_root))

import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy
from tqdm import tqdm


# Organ class configuration
ORGAN_CLASSES = {
    'colon': ['Budding_opaque', 'Budding_transparent', 'Non_budding_opaque', 'Non_budding_transparent'],
    'lung': ['lumen', 'non_lumen'],
    'stomach': ['Apoptotic', 'Hollow', 'Solid'],
}

# Default model paths
MODEL_PATHS = {
    'colon':   'checkpoints/training-runs/00003-Colon64-gpus1-batch64-colon64/network-snapshot-latest.pkl',
    'lung':    'checkpoints/training-runs/00005-Lung64-gpus1-batch64-lung64/network-snapshot-latest.pkl',
    'stomach': 'checkpoints/training-runs/00004-Stomach64-gpus1-batch64-stomach64/network-snapshot-latest.pkl',
}


def load_model(model_path: str, device: torch.device):
    """Load an R3GAN generator from a pickle checkpoint."""
    print(f"Loading model from: {model_path}")
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print(f"  z_dim: {G.z_dim}, c_dim: {G.c_dim}, resolution: {G.img_resolution}")
    return G


def generate_images_for_organ(
    G,
    organ_name: str,
    class_names: List[str],
    runs: int,
    images_per_class: int,
    outdir: Path,
    device: torch.device,
    start_seed: int = 0,
):
    """
    Generate images for all classes of a single organ.

    Args:
        G: R3GAN generator.
        organ_name: Name of the organ (used for output sub-directory).
        class_names: List of class names.
        runs: Number of generation runs.
        images_per_class: Images generated per class per run.
        outdir: Root output directory.
        device: Torch device.
        start_seed: Starting random seed.
    """
    organ_dir = outdir / organ_name
    organ_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)

    # Validate that model c_dim matches the number of classes
    if G.c_dim != num_classes:
        print(f"  WARNING: Model c_dim={G.c_dim}, but {organ_name} has {num_classes} classes!")
        print(f"  Proceeding with {min(G.c_dim, num_classes)} classes...")
        num_classes = min(G.c_dim, num_classes)

    print(f"\n{'='*60}")
    print(f"Generating images for: {organ_name}")
    print(f"{'='*60}")
    print(f"  Output directory: {organ_dir}")
    print(f"  Classes: {', '.join(class_names[:num_classes])}")
    print(f"  Runs: {runs}")
    print(f"  Images per class per run: {images_per_class}")
    print(f"  Total images per class: {runs * images_per_class}")
    print(f"  Grand total: {runs * images_per_class * num_classes}")

    for run_idx in range(1, runs + 1):
        run_dir = organ_dir / f"run{run_idx}"
        run_dir.mkdir(exist_ok=True)

        print(f"\n  Run {run_idx}/{runs}")

        for class_idx, class_name in enumerate(class_names[:num_classes]):
            class_dir = run_dir / class_name
            class_dir.mkdir(exist_ok=True)

            # Build one-hot class label
            label = torch.zeros([1, G.c_dim], device=device)
            label[:, class_idx] = 1

            print(f"    {class_name} (class_idx={class_idx})")

            # Each class/run uses a distinct seed range to avoid overlap
            seed_offset = (run_idx - 1) * images_per_class * num_classes + class_idx * images_per_class
            current_seed = start_seed + seed_offset

            for img_idx in tqdm(range(images_per_class), desc=f"      Generating", leave=False):
                seed = current_seed + img_idx

                # Sample random noise
                z = torch.from_numpy(
                    np.random.RandomState(seed).randn(1, G.z_dim)
                ).to(device)

                # Generate image
                with torch.no_grad():
                    img = G(z, label)

                # Denormalise and convert to uint8
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                img_path = class_dir / f"sample_{img_idx:04d}.png"
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(img_path)

    print(f"\n  Completed {organ_name}!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate organoid images by organ types and classes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all organs, 10 runs, 2000 images per class per run
  python scripts/generate_organoid_images.py --runs 10 --images_per_class 2000 --outdir outputs/Orgdif

  # Generate only for stomach, 5 runs, 1000 images per class per run
  python scripts/generate_organoid_images.py --organ stomach --runs 5 --images_per_class 1000 --outdir outputs/stomach_test
        """
    )

    # Core arguments
    parser.add_argument('--runs', type=int, required=True,
                        help='Number of generation runs')
    parser.add_argument('--images_per_class', type=int, required=True,
                        help='Number of images per class per run')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')

    # Organ selection
    parser.add_argument('--organ', type=str, default='all',
                        choices=['all', 'stomach', 'colon', 'lung'],
                        help='Which organ to generate (default: all)')

    # Model paths (optional overrides)
    parser.add_argument('--stomach_model', type=str,
                        default='checkpoints/training-runs/00004-Stomach64-gpus1-batch64-stomach64/network-snapshot-latest.pkl',
                        help='Path to Stomach R3GAN checkpoint')
    parser.add_argument('--colon_model', type=str,
                        default='checkpoints/training-runs/00003-Colon64-gpus1-batch64-colon64/network-snapshot-latest.pkl',
                        help='Path to Colon R3GAN checkpoint')
    parser.add_argument('--lung_model', type=str,
                        default='checkpoints/training-runs/00005-Lung64-gpus1-batch64-lung64/network-snapshot-latest.pkl',
                        help='Path to Lung R3GAN checkpoint')

    # Miscellaneous
    parser.add_argument('--start_seed', type=int, default=0,
                        help='Starting seed (default: 0)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')

    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine which organs to generate
    organs_config = {
        'stomach': args.stomach_model,
        'colon': args.colon_model,
        'lung': args.lung_model,
    }

    if args.organ != 'all':
        if args.organ in organs_config:
            organs_config = {args.organ: organs_config[args.organ]}
        else:
            raise ValueError(f"Unknown organ: {args.organ}")

    # Output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Print configuration summary
    print("\n" + "="*60)
    print("Organoid Image Generation")
    print("="*60)
    print(f"Output directory: {outdir}")
    print(f"Organs: {', '.join(organs_config.keys())}")
    print(f"Runs: {args.runs}")
    print(f"Images per class per run: {args.images_per_class}")

    # Compute totals
    total_images = 0
    for organ_name in organs_config.keys():
        num_classes = len(ORGAN_CLASSES[organ_name])
        organ_total = args.runs * args.images_per_class * num_classes
        print(f"  {organ_name}: {num_classes} classes × {args.runs} runs × {args.images_per_class} images = {organ_total} images")
        total_images += organ_total

    print(f"Grand total: {total_images} images")
    print("="*60)

    # Generate images for each organ
    for organ_name, model_path in organs_config.items():
        if not Path(model_path).exists():
            print(f"\nError: Model not found: {model_path}")
            print(f"Skipping {organ_name}...")
            continue

        G = load_model(model_path, device)

        generate_images_for_organ(
            G=G,
            organ_name=organ_name,
            class_names=ORGAN_CLASSES[organ_name],
            runs=args.runs,
            images_per_class=args.images_per_class,
            outdir=outdir,
            device=device,
            start_seed=args.start_seed,
        )

    print("\n" + "="*60)
    print("All generation completed!")
    print(f"Results saved to: {outdir}")
    print("="*60)

    # Print example directory structure
    print("\nDirectory structure:")
    for organ_name in organs_config.keys():
        print(f"  {outdir}/{organ_name}/")
        print(f"    run1/")
        for class_name in ORGAN_CLASSES[organ_name][:2]:
            print(f"      {class_name}/")
            print(f"        sample_0000.png ~ sample_{args.images_per_class-1:04d}.png")
        if len(ORGAN_CLASSES[organ_name]) > 2:
            print(f"      ... ({len(ORGAN_CLASSES[organ_name]) - 2} more classes)")
        print(f"    run2/")
        print(f"      ...")
        print(f"    run{args.runs}/")


if __name__ == '__main__':
    main()
