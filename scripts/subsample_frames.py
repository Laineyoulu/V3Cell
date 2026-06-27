#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video frame subsampling script

Subsample frames by taking every N-th frame to reduce total frame count.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def subsample_video_dataset(
    src_dir: str,
    dst_dir: str,
    step: int = 2,
    frame_pattern: str = "*_t*.png"
):
    """
    Subsample video dataset frames

    Args:
        src_dir: Source directory (e.g., Dataset/vedio)
        dst_dir: Destination directory (e.g., Dataset/vedio_half)
        step: Sampling step (2 means take every 2nd frame)
        frame_pattern: Frame file pattern
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # Create destination directory
    dst_path.mkdir(parents=True, exist_ok=True)

    # Statistics
    total_videos = 0
    total_src_frames = 0
    total_dst_frames = 0

    # Iterate class directories (lumen, lumen_collapse, non_lumen)
    for class_dir in sorted(src_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        dst_class_dir = dst_path / class_name
        dst_class_dir.mkdir(exist_ok=True)

        print(f"\nProcessing class: {class_name}")

        # Iterate video directories
        video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])

        for video_dir in tqdm(video_dirs, desc=f"  {class_name}"):
            video_name = video_dir.name
            dst_video_dir = dst_class_dir / video_name
            dst_video_dir.mkdir(exist_ok=True)

            # Get all frame files and sort
            frame_files = sorted(video_dir.glob(frame_pattern))

            if len(frame_files) == 0:
                continue

            total_videos += 1
            total_src_frames += len(frame_files)

            # Take every N-th frame
            sampled_frames = frame_files[::step]
            total_dst_frames += len(sampled_frames)

            # Copy and rename
            # Extract prefix from original filename (e.g., "lumen_1" from "lumen_1_t000.png")
            first_frame_name = frame_files[0].name
            prefix = first_frame_name.rsplit('_t', 1)[0]

            for new_idx, src_frame in enumerate(sampled_frames):
                # New filename: prefix_t{new_idx:03d}.png
                new_name = f"{prefix}_t{new_idx:03d}.png"
                dst_frame = dst_video_dir / new_name
                shutil.copy2(src_frame, dst_frame)

    print(f"\n{'='*50}")
    print(f"Subsampling complete!")
    print(f"  Source dir: {src_dir}")
    print(f"  Dest dir: {dst_dir}")
    print(f"  Step: every {step} frames")
    print(f"  Total videos: {total_videos}")
    print(f"  Original frames: {total_src_frames}")
    print(f"  Subsampled frames: {total_dst_frames}")
    print(f"  Compression ratio: {total_src_frames / total_dst_frames:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Subsample video frames')
    parser.add_argument('--src', type=str, default='Dataset/vedio',
                        help='Source video directory')
    parser.add_argument('--dst', type=str, default='Dataset/vedio_half',
                        help='Destination directory')
    parser.add_argument('--step', type=int, default=2,
                        help='Sampling step (2=take every 2nd frame)')

    args = parser.parse_args()

    subsample_video_dataset(
        src_dir=args.src,
        dst_dir=args.dst,
        step=args.step,
    )
