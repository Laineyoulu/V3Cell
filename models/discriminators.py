"""
Discriminator networks for V3Cell (MoCoGAN-HD specification).

Two discriminators operate in parallel during adversarial training:
  - ImageDiscriminator  (2D): Receives a pair [frame_0, frame_t] concatenated
      along the channel axis.  Learns to distinguish real and generated frame
      pairs, which implicitly captures temporal coherence.
  - VideoDiscriminator  (3D): Multi-scale 3D conv discriminator that processes
      the full video volume.  Following MoCoGAN-HD, the input is the difference
      representation [frame_0 repeated, frame_1, …, frame_T-1] to help the
      discriminator focus on motion rather than absolute appearance.

Both discriminators carry their own Adam optimisers as attributes so that
the training loop can call `disc.optim.step()` without managing separate
optimizer dictionaries.
"""

import numpy as np
import functools
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def weights_init(m: nn.Module):
    """MoCoGAN-HD style weight initialisation (N(0, 0.02) for conv, N(1, 0.02) for BN)."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer_2d(norm_type: str = 'instance'):
    """Return a 2D normalisation layer factory for the given norm_type."""
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        return None
    raise NotImplementedError(f"Normalisation layer '{norm_type}' is not supported.")


def get_norm_layer_3d(norm_type: str = 'instance'):
    """Return a 3D normalisation layer factory for the given norm_type."""
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        return None
    raise NotImplementedError(f"Normalisation layer '{norm_type}' is not supported.")


# =============================================================================
# 2D Image Discriminator
# =============================================================================

class ImageDiscriminator(nn.Module):
    """
    2D frame-pair discriminator (MoCoGAN-HD style).

    Accepts the concatenation of the first frame and a randomly sampled frame
    [frame_0 | frame_t] along the channel axis.  This forces the discriminator
    to reason about temporal relationships between frames rather than treating
    each frame independently.

    Architecture: PatchGAN (N-layer) with stride-2 convolutions.
    """

    def __init__(
        self,
        img_resolution: int = 64,
        img_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = 'instance',
        use_sigmoid: bool = False,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        """
        Args:
            img_resolution: Spatial resolution of each input frame.
            img_channels:   Channels per frame (doubled internally for the pair input).
            ndf:            Base number of discriminator feature maps.
            n_layers:       Number of strided conv layers (depth of PatchGAN).
            norm_type:      Normalisation type: 'instance', 'batch', or 'none'.
            use_sigmoid:    If True, add a final sigmoid (not needed for LSGAN).
            lr:             Adam learning rate.
            beta1, beta2:   Adam β parameters.
        """
        super().__init__()

        self.img_resolution = img_resolution
        input_nc = img_channels * 2  # concatenated frame pair

        norm_layer = get_norm_layer_2d(norm_type)
        kw, padw = 4, 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            block = [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
            if norm_layer is not None:
                block.insert(1, norm_layer(ndf * nf_mult))
            sequence += block

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        block = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        if norm_layer is not None:
            block.insert(1, norm_layer(ndf * nf_mult))
        sequence += block

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.model.apply(weights_init)

        self.optim = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C*2, H, W] — concatenated frame pair (frame_0 | frame_t).

        Returns:
            logits: PatchGAN output map.
        """
        return self.model(x)


# =============================================================================
# 3D Video Discriminator
# =============================================================================

class NLayerDiscriminator3D(nn.Module):
    """
    Single-scale 3D discriminator with optional intermediate feature extraction.

    When getIntermFeat=True, each layer is stored as a separate sub-module
    (model0, model1, …) so that the caller can access intermediate feature maps
    for the feature-matching loss.
    """

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer=nn.InstanceNorm3d,
        getIntermFeat: bool = True,
    ):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        # Build layer blocks as a list-of-lists for easy per-layer storage.
        sequence = [[
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True),
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n, block in enumerate(sequence):
                setattr(self, f'model{n}', nn.Sequential(*block))
        else:
            self.model = nn.Sequential(*[layer for block in sequence for layer in block])

    def forward(self, input: torch.Tensor):
        if self.getIntermFeat:
            # Return a list of per-layer activations for feature matching.
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, f'model{n}')
                res.append(model(res[-1]))
            return res[1:]
        return self.model(input)


class VideoDiscriminator(nn.Module):
    """
    Multi-scale 3D video discriminator (MoCoGAN-HD style).

    Stacks multiple NLayerDiscriminator3D instances operating at different
    spatial scales.  The input is progressively downsampled between scales
    via average pooling, allowing each discriminator to focus on different
    temporal frequency ranges.

    Input format: [B, C*2, T-1, H', W'] (see prepare_video_disc_input).
    """

    def __init__(
        self,
        img_channels: int = 3,
        n_frames: int = 8,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = 'instance',
        num_D: int = 2,
        getIntermFeat: bool = True,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        """
        Args:
            img_channels:  Channels per frame (doubled internally).
            n_frames:      Number of frames in the video clip.
            ndf:           Base feature map count for the coarsest discriminator.
            n_layers:      Conv layers per discriminator scale.
            norm_type:     Normalisation type.
            num_D:         Number of discriminator scales.
            getIntermFeat: Whether to return intermediate features (for FM loss).
            lr:            Adam learning rate.
            beta1, beta2:  Adam β parameters.
        """
        super().__init__()

        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.n_frames = n_frames

        input_nc = img_channels * 2  # [frame_0 repeated | frame_t]
        norm_layer = get_norm_layer_3d(norm_type)
        ndf_max = 64

        for i in range(num_D):
            netD = NLayerDiscriminator3D(
                input_nc=input_nc,
                ndf=min(ndf_max, ndf * (2 ** (num_D - 1 - i))),
                n_layers=n_layers,
                norm_layer=norm_layer,
                getIntermFeat=getIntermFeat,
            )
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, f'scale{i}_layer{j}', getattr(netD, f'model{j}'))
            else:
                setattr(self, f'layer{i}', netD.model)

        # Temporal downsampling for multi-scale: stride along temporal axis only
        # when the clip is short (<=16 frames), otherwise full 3D average pooling.
        if n_frames > 16:
            self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1],
                                           count_include_pad=False)
        else:
            self.downsample = nn.AvgPool3d(3, stride=[1, 2, 2], padding=[1, 1, 1],
                                           count_include_pad=False)

        self.apply(weights_init)
        self.optim = optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2))

    def singleD_forward(self, model, input: torch.Tensor) -> list:
        """Run a single discriminator scale and return intermediate features."""
        if self.getIntermFeat:
            result = [input]
            for layer in model:
                result.append(layer(result[-1]))
            return result[1:]
        return [model(input)]

    def forward(self, input: torch.Tensor) -> List:
        """
        Args:
            input: [B, C*2, T-1, H', W'] — video in MoCoGAN-HD difference format.

        Returns:
            result: List of per-scale discriminator outputs.
                    Each element is a list of intermediate feature maps.
        """
        num_D = self.num_D
        result = []
        input_downsampled = input

        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self, f'scale{num_D - 1 - i}_layer{j}')
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, f'layer{num_D - 1 - i}')

            result.append(self.singleD_forward(model, input_downsampled))

            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)

        return result


# =============================================================================
# Input preparation utilities
# =============================================================================

def prepare_image_disc_input(video: torch.Tensor, frame_id: int) -> torch.Tensor:
    """
    Prepare input for the 2D ImageDiscriminator.

    Concatenates the first frame with a chosen target frame along the channel
    axis, matching the format expected by ImageDiscriminator.

    Args:
        video:    [B, T, C, H, W] — video tensor.
        frame_id: Index of the target frame to pair with the first frame.

    Returns:
        input: [B, C*2, H, W]
    """
    first_frame = video[:, 0]       # [B, C, H, W]
    target_frame = video[:, frame_id]  # [B, C, H, W]
    return torch.cat([first_frame, target_frame], dim=1)


def prepare_video_disc_input(
    video: torch.Tensor,
    target_spatial_size: int = 128,
) -> torch.Tensor:
    """
    Prepare input for the 3D VideoDiscriminator (MoCoGAN-HD difference format).

    Constructs a difference volume where each temporal slice t is the
    concatenation [frame_0, frame_t].  Spatial resolution is downsampled
    to target_spatial_size to reduce discriminator input size.

    Layout:
        Input:  [B, T, C, H, W]
        Output: [B, C*2, T-1, H', W']   where H' = W' = target_spatial_size

    Args:
        video:               [B, T, C, H, W]
        target_spatial_size: Target H and W after downsampling (default 128).

    Returns:
        combined: [B, C*2, T-1, H', W']
    """
    B, T, C, H, W = video.shape

    # Repeat the first frame T-1 times for the anchor channel.
    first_frames = video[:, 0:1].repeat(1, T - 1, 1, 1, 1)  # [B, T-1, C, H, W]
    rest_frames = video[:, 1:]                                 # [B, T-1, C, H, W]

    # Channel-wise concatenation → temporal axis.
    combined = torch.cat([first_frames, rest_frames], dim=2)  # [B, T-1, C*2, H, W]

    # Reshape to [B, C*2, T-1, H, W] for 3D conv input.
    combined = combined.permute(0, 2, 1, 3, 4)

    # Spatially downsample to target_spatial_size (if needed).
    if H > target_spatial_size or W > target_spatial_size:
        scale = H // target_spatial_size
        if scale > 1:
            combined = F.avg_pool3d(
                combined,
                kernel_size=(1, scale, scale),
                stride=(1, scale, scale),
            )

    return combined
