"""
Dataset with foreground mask support (for fine-tuning).

Extends the original ImageFolderDataset by adding mask loading functionality.
The original dataset code is left completely unchanged.
"""

import os
import numpy as np
import zipfile
import PIL.Image
import json
from training.dataset import ImageFolderDataset

try:
    import pyspng
except ImportError:
    pyspng = None


class ImageFolderDatasetWithMask(ImageFolderDataset):
    """
    Extended dataset that loads per-image foreground masks alongside images.

    Masks can come from:
    1. A 'masks' sub-directory inside the same zip/directory as the images.
    2. A separate mask zip file.
    3. A separate mask directory.
    """

    def __init__(self,
        path,                   # Path to directory or zip.
        mask_path=None,         # Path to mask directory or zip. None = look in same location as images.
        mask_subdir='masks',    # Subdirectory name for masks within zip/dir.
        mask_suffix='',         # Suffix to add to image filename to get mask filename.
        resolution=None,        # Ensure specific resolution.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.mask_path = mask_path
        self.mask_subdir = mask_subdir
        self.mask_suffix = mask_suffix
        self._mask_zipfile = None

        # Initialise parent class
        super().__init__(path=path, resolution=resolution, **super_kwargs)

        # Verify that masks can be loaded
        self._verify_masks()

    def _verify_masks(self):
        """Try to load the first mask to verify availability."""
        try:
            mask = self._load_mask(0)
            if mask is not None:
                print(f"Mask files detected successfully. Shape: {mask.shape}")
            else:
                print("Warning: No mask files found. Training will proceed without masks.")
        except Exception as e:
            print(f"Warning: Failed to load mask for the first image: {e}")
            print("   Training will proceed without masks.")

    def _get_mask_zipfile(self):
        """Open (and cache) the mask zip file handle."""
        if self._mask_zipfile is None and self.mask_path is not None:
            if self.mask_path.endswith('.zip') and os.path.exists(self.mask_path):
                self._mask_zipfile = zipfile.ZipFile(self.mask_path)
        return self._mask_zipfile

    def _open_mask_file(self, fname, use_external_path=False):
        """Open a mask file and return a file-like object, or None if not found."""
        if use_external_path and self.mask_path is not None:
            if os.path.isdir(self.mask_path):
                return open(os.path.join(self.mask_path, fname), 'rb')
            elif self.mask_path.endswith('.zip'):
                zf = self._get_mask_zipfile()
                if zf is not None:
                    return zf.open(fname, 'r')

        # Fallback: try to find in the same location as images
        if self._type == 'dir':
            mask_dir = os.path.join(self._path, self.mask_subdir)
            if os.path.isdir(mask_dir):
                return open(os.path.join(mask_dir, fname), 'rb')
        elif self._type == 'zip':
            mask_fname = f"{self.mask_subdir}/{fname}"
            if mask_fname in self._all_fnames:
                return self._get_zipfile().open(mask_fname, 'r')

        return None

    def _load_mask(self, raw_idx):
        """Load the foreground mask for the image at raw_idx."""
        # Derive mask filename from image filename
        img_fname = self._image_fnames[raw_idx]
        img_basename = os.path.basename(img_fname)
        img_name, img_ext = os.path.splitext(img_basename)

        mask_fname = f"{img_name}{self.mask_suffix}.png"

        # Try external mask path first
        if self.mask_path is not None:
            try:
                f = self._open_mask_file(mask_fname, use_external_path=True)
                if f is not None:
                    with f:
                        mask = np.array(PIL.Image.open(f).convert('L'))
                        mask = mask.astype(np.float32) / 255.0  # Normalise to [0, 1]
                        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                        mask = mask[np.newaxis, :, :]  # HW -> 1HW
                        return mask
            except Exception:
                pass

        # Try same location as images
        try:
            f = self._open_mask_file(mask_fname, use_external_path=False)
            if f is not None:
                with f:
                    mask = np.array(PIL.Image.open(f).convert('L'))
                    mask = mask.astype(np.float32) / 255.0
                    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
                    mask = mask[np.newaxis, :, :]
                    return mask
        except Exception:
            pass

        # Fallback: all-ones mask (equivalent to no masking)
        return np.ones([1, self.resolution, self.resolution], dtype=np.float32)

    def __getitem__(self, idx):
        """Return (image, label, mask)."""
        image, label = super().__getitem__(idx)

        raw_idx = self._raw_idx[idx]
        mask = self._load_mask(raw_idx)

        # Apply the same horizontal flip as the image if needed
        if self._xflip[idx]:
            mask = mask[:, :, ::-1].copy()

        return image, label, mask

    def close(self):
        """Close all open file handles."""
        try:
            if self._mask_zipfile is not None:
                self._mask_zipfile.close()
        finally:
            self._mask_zipfile = None
        super().close()

    def __getstate__(self):
        return dict(super().__getstate__(), _mask_zipfile=None)
