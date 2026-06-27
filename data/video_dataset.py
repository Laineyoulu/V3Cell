# -*- coding: utf-8 -*-
"""
Video dataset for V3Cell.

Loads organoid development videos from a directory of image sequences.
Supports two directory layouts:

  Flat layout (one level):
    video_dir/
    ├── video_001/
    │   ├── frame_t000.png
    │   ├── frame_t001.png
    │   └── …
    └── video_002/

  Nested layout (class / video / frames):
    video_dir/
    ├── lumen/
    │   ├── video_001/
    │   │   ├── frame_t000.png
    │   │   └── …
    │   └── video_002/
    ├── lumen_collapse/
    └── non_lumen/

Class labels are inferred from the parent directory name when using the nested
layout, or loaded from an optional labels.json file.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class VideoDataset(Dataset):
    """
    Organoid development video dataset.

    Each sample is a temporally contiguous clip of n_frames frames drawn from a
    single video.  Clips are optionally randomly subsampled along the time axis
    when the video has more frames than n_frames.

    Returned dictionary keys:
      'video'     : [T, C, H, W] float tensor in [-1, 1].
      'label'     : [c_dim] one-hot class label.
      'video_idx' : integer index into the video list.
    """

    # Hard-coded class mapping used when labels are inferred from directory names.
    CLASS_TO_LABEL: Dict[str, int] = {
        'lumen': 0,
        'lumen_collapse': 1,
        'non_lumen': 2,
    }

    def __init__(
        self,
        video_dir: Union[str, Path],
        n_frames: int = 8,
        img_resolution: int = 64,
        frame_pattern: str = "frame_*.png",
        min_frames: int = 8,
        max_frames: Optional[int] = None,
        c_dim: int = 4,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        random_start: bool = True,
        labels_file: Optional[str] = None,
    ):
        """
        Args:
            video_dir:        Root directory containing video sub-folders.
            n_frames:         Number of frames per returned clip.
            img_resolution:   Resize each frame to this square resolution.
            frame_pattern:    Glob pattern for frame files (e.g. '*_t*.png').
            min_frames:       Minimum frames required to include a video.
            max_frames:       Cap on the number of frames loaded per video.
            c_dim:            Dimension of the one-hot class label vector.
            horizontal_flip:  Apply random horizontal flipping.
            vertical_flip:    Apply random vertical flipping.
            random_start:     Randomly choose the clip start frame; otherwise use 0.
            labels_file:      Name of a JSON label file inside video_dir.
        """
        super().__init__()

        self.video_dir = Path(video_dir)
        self.n_frames = n_frames
        self.img_resolution = img_resolution
        self.frame_pattern = frame_pattern
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.c_dim = c_dim
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_start = random_start

        # Load optional per-video label overrides.
        self.labels: Dict[str, int] = {}
        if labels_file is not None:
            labels_path = self.video_dir / labels_file
            if labels_path.exists():
                with open(labels_path) as f:
                    self.labels = json.load(f)
        elif (self.video_dir / 'labels.json').exists():
            with open(self.video_dir / 'labels.json') as f:
                self.labels = json.load(f)

        self.videos = self._scan_videos()

        if len(self.videos) == 0:
            raise ValueError(f"No valid videos found in {video_dir}")

        print(f"Found {len(self.videos)} videos with >= {min_frames} frames")

    def _scan_videos(self) -> List[Dict]:
        """
        Recursively scan the dataset directory for video clips.

        Handles both flat and nested (class/video/frames) directory layouts.
        """
        videos = []

        for item_path in sorted(self.video_dir.iterdir()):
            if not item_path.is_dir():
                continue

            frame_files = sorted(item_path.glob(self.frame_pattern))

            if len(frame_files) >= self.min_frames:
                # Flat layout: item_path is a video directory.
                if self.max_frames is not None and len(frame_files) > self.max_frames:
                    frame_files = frame_files[:self.max_frames]

                video_name = item_path.name
                label = self.labels.get(video_name, 0)

                videos.append({
                    'name': video_name,
                    'path': item_path,
                    'frames': frame_files,
                    'n_frames': len(frame_files),
                    'label': label,
                })
            else:
                # Nested layout: item_path is a class directory.
                class_name = item_path.name
                class_label = self.CLASS_TO_LABEL.get(class_name, 0)

                for video_path in sorted(item_path.iterdir()):
                    if not video_path.is_dir():
                        continue

                    frame_files = sorted(video_path.glob(self.frame_pattern))

                    if len(frame_files) < self.min_frames:
                        continue

                    if self.max_frames is not None and len(frame_files) > self.max_frames:
                        frame_files = frame_files[:self.max_frames]

                    video_name = f"{class_name}/{video_path.name}"
                    label = self.labels.get(video_name, class_label)

                    videos.append({
                        'name': video_name,
                        'path': video_path,
                        'frames': frame_files,
                        'n_frames': len(frame_files),
                        'label': label,
                    })

        return videos

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a video clip.

        Returns:
            dict with keys:
              'video':     [T, C, H, W] float tensor in [-1, 1].
              'label':     [c_dim] one-hot label.
              'video_idx': int index.
        """
        video_info = self.videos[idx]
        frame_files = video_info['frames']
        n_total = video_info['n_frames']
        label = video_info['label']

        # Choose the starting frame for this clip.
        if self.random_start and n_total > self.n_frames:
            start_idx = random.randint(0, n_total - self.n_frames)
        else:
            start_idx = 0

        # Load and resize frames.
        frames = []
        for i in range(self.n_frames):
            frame_idx = min(start_idx + i, n_total - 1)
            img = Image.open(frame_files[frame_idx]).convert('RGB')
            if img.size != (self.img_resolution, self.img_resolution):
                img = img.resize(
                    (self.img_resolution, self.img_resolution), Image.BILINEAR
                )
            frames.append(np.array(img))

        frames = np.stack(frames, axis=0)  # [T, H, W, C]

        # Data augmentation.
        if self.horizontal_flip and random.random() > 0.5:
            frames = frames[:, :, ::-1, :]
        if self.vertical_flip and random.random() > 0.5:
            frames = frames[:, ::-1, :, :]

        # Convert to tensor and normalise to [-1, 1].
        frames = torch.from_numpy(frames.copy()).float()
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = frames / 127.5 - 1.0

        # Build one-hot class label.
        c = torch.zeros(self.c_dim)
        if 0 <= label < self.c_dim:
            c[label] = 1.0
        else:
            c[0] = 1.0  # default to first class

        return {'video': frames, 'label': c, 'video_idx': idx}


class VideoFrameDataset(Dataset):
    """
    Single-frame dataset for image-level (Stage 1) training.

    Flattens all video directories into an enumerable list of individual frames.
    Each sample is a single RGB image with its associated class label.

    Returned dictionary keys:
      'image': [C, H, W] float tensor in [-1, 1].
      'label': [c_dim] one-hot class label.
    """

    def __init__(
        self,
        video_dir: Union[str, Path],
        img_resolution: int = 64,
        frame_pattern: str = "frame_*.png",
        c_dim: int = 4,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        labels_file: Optional[str] = None,
    ):
        super().__init__()

        self.video_dir = Path(video_dir)
        self.img_resolution = img_resolution
        self.frame_pattern = frame_pattern
        self.c_dim = c_dim
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.labels: Dict[str, int] = {}
        if labels_file is not None:
            labels_path = self.video_dir / labels_file
            if labels_path.exists():
                with open(labels_path) as f:
                    self.labels = json.load(f)
        elif (self.video_dir / 'labels.json').exists():
            with open(self.video_dir / 'labels.json') as f:
                self.labels = json.load(f)

        self.frames = self._scan_frames()

        if len(self.frames) == 0:
            raise ValueError(f"No frames found in {video_dir}")

        print(f"Found {len(self.frames)} frames from video dataset")

    def _scan_frames(self) -> List[Dict]:
        """Enumerate all frame files across all video directories."""
        frames = []
        for video_path in sorted(self.video_dir.iterdir()):
            if not video_path.is_dir():
                continue
            video_name = video_path.name
            label = self.labels.get(video_name, 0)
            for frame_file in sorted(video_path.glob(self.frame_pattern)):
                frames.append({
                    'path': frame_file,
                    'video_name': video_name,
                    'label': label,
                })
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_info = self.frames[idx]
        label = frame_info['label']

        img = Image.open(frame_info['path']).convert('RGB')
        if img.size != (self.img_resolution, self.img_resolution):
            img = img.resize((self.img_resolution, self.img_resolution), Image.BILINEAR)
        img = np.array(img)

        if self.horizontal_flip and random.random() > 0.5:
            img = img[:, ::-1, :]
        if self.vertical_flip and random.random() > 0.5:
            img = img[::-1, :, :]

        img = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 127.5 - 1.0

        c = torch.zeros(self.c_dim)
        if 0 <= label < self.c_dim:
            c[label] = 1.0
        else:
            c[0] = 1.0

        return {'image': img, 'label': c}


def create_video_dataloader(
    video_dir: Union[str, Path],
    batch_size: int = 16,
    n_frames: int = 8,
    img_resolution: int = 64,
    c_dim: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Convenience factory for a VideoDataset DataLoader.

    Args:
        video_dir:      Root video directory.
        batch_size:     Batch size.
        n_frames:       Frames per clip.
        img_resolution: Frame spatial resolution.
        c_dim:          Class label dimension.
        num_workers:    DataLoader worker processes.
        pin_memory:     Pin CPU memory for faster GPU transfer.
        **kwargs:       Additional arguments forwarded to VideoDataset.

    Returns:
        DataLoader with shuffle=True and drop_last=True.
    """
    dataset = VideoDataset(
        video_dir=video_dir,
        n_frames=n_frames,
        img_resolution=img_resolution,
        c_dim=c_dim,
        **kwargs,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
