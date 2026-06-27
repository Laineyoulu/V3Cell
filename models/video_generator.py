# -*- coding: utf-8 -*-
"""
V3CellGenerator: Main video generation model for V3Cell.

Encapsulates a frozen temporal organoid generator (an R3GAN-architecture
image generator trained in Stage 1 and frozen for Stage 2; referred to as
G_AS in the paper's amniotic-sac experiments — the generator itself is
lineage-agnostic), a MotionRNN for temporal latent trajectory prediction,
and a SequenceEncoder for conditioning on early frames.

Two operating modes:
  1. Unconditional: sample z ~ N(0,I), generate full video via MotionRNN.
  2. Conditional:  given early observed frames, encode them, warm-up LSTM hidden
                   state, then autoregressively predict future latents and decode
                   each latent to a frame through the frozen generator.

Architecture (panel c of the paper):
  - Static Modeling (frozen):  z, mask, label → frozen generator → frame
  - Dynamic Modeling:
      early_frames → SequenceEncoder → (z_seq, h_k, c_k)
                                            ↓
      z_t = z_c ⊕ Δz_t  (add noise eps_t via MotionRNN LSTM)
                                            ↓
      frozen generator → predicted frame X̃_t  → temporal virtual organoid (video)
"""

import sys
import pickle
import importlib
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from .motion_rnn import MotionRNN, MotionRNNWithPCA, ContentMotionRNN
from .encoder import SequenceEncoder, ImageEncoder
from .fate_classifier import FateClassifier
from .temporal_organoid_generator.networks import Generator as TemporalOrganoidGenerator


class V3CellGenerator(nn.Module):
    """
    V3Cell video generator.

    Combines a frozen temporal organoid generator with a trainable MotionRNN
    and SequenceEncoder to produce temporally coherent organoid development videos.

    Training strategy:
    - The frozen generator's parameters are frozen — only MotionRNN and Encoder
      are trained.
    - Gradients flow from the generated frames back through the frozen generator
      (no_grad is NOT applied to the forward pass) to train the Encoder and
      MotionRNN. This works because the frozen generator's parameters have
      requires_grad=False, so they consume no memory for optimizer state but
      still allow gradient propagation through their activations.

    Forward pipeline:
        z ~ N(0,I)  ──► MotionRNN ──► [z_0, …, z_T] ──► frozen generator ──► frames

    Conditional pipeline (predict_from_frames):
        early_frames ──► SequenceEncoder ──► z_seq_input, (h_k, c_k)
                                               ──► MotionRNN.forward_from_sequence
                                               ──► z_seq_full ──► frozen generator ──► video
    """

    def __init__(
        self,
        r3gan_checkpoint: Union[str, Path],
        z_dim: int = 64,
        c_dim: int = 3,
        img_resolution: int = 256,
        motion_hidden_dim: int = 384,   # Following MoCoGAN-HD: 384
        motion_num_layers: int = 2,
        motion_dropout: float = 0.1,
        residual_weight: float = 0.1,
        use_pca: bool = False,
        n_pca: int = 32,
        use_content_motion: bool = False,  # MoCoGAN-style content-motion separation
        content_dim: int = 32,
        use_encoder: bool = True,
        encoder_base_channels: int = 64,
        # --- Fate Classifier (Phi in the paper) ---
        use_fate_classifier: bool = False,
        fate_classifier_base_channels: int = 32,
        fate_classifier_feat_dim: int = 256,
        fate_classifier_hidden_dim: int = 128,
        fate_classifier_temperature: float = 1.0,
        # ----------------------
        freeze_r3gan: bool = True,
        r3gan_path: Optional[str] = None,
        device_ids: Optional[list] = None,
    ):
        """
        Args:
            r3gan_checkpoint: Path to the pre-trained R3GAN .pkl checkpoint
                               (used here to instantiate the frozen generator
                               generator).
            z_dim: Latent noise dimension (R3GAN uses NoiseDimension=64).
            c_dim: Condition (class label) dimension.
            img_resolution: Spatial resolution of generated frames.
            motion_hidden_dim: LSTM hidden state size in MotionRNN (MoCoGAN-HD: 384).
            motion_num_layers: Number of LSTM layers in MotionRNN.
            motion_dropout: Dropout probability inside MotionRNN LSTM.
            residual_weight: Scale factor for the residual latent update Δz.
            use_pca: If True, use MotionRNNWithPCA (learn motion in PCA subspace).
            n_pca: Number of PCA components when use_pca=True.
            use_content_motion: If True, use MoCoGAN content-motion separation.
                                 z = [z_C (fixed per video), z_M (time-varying)].
            content_dim: Dimensionality of the content code z_C.
            use_encoder: If True, attach a SequenceEncoder for conditional generation.
            encoder_base_channels: Base channel width of the ResNet encoder backbone.
            use_fate_classifier: If True, attach a FateClassifier (Phi) that infers
                                 class C from the early frames rather than using
                                 ground-truth labels. This implements the
                                 'Fate Predict' block from Figure c.
            fate_classifier_base_channels: CNN base channel width for FateClassifier.
            fate_classifier_feat_dim: Per-frame feature dimension after CNN encoding.
            fate_classifier_hidden_dim: Bidirectional LSTM hidden size per direction.
            fate_classifier_temperature: Softmax temperature τ (1.0 = standard softmax).
            freeze_r3gan: If True, freeze all frozen-generator parameters (recommended).
            r3gan_path: Optional path to external R3GAN source (unused in most setups).
            device_ids: List of GPU ids for DataParallel wrapping of the frozen generator.
        """
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.freeze_r3gan = freeze_r3gan
        self.use_encoder = use_encoder
        self.use_fate_classifier = use_fate_classifier
        self.device_ids = device_ids

        # --- Load the frozen generator ---
        self.frozen_generator = self._load_frozen_generator(r3gan_checkpoint)

        if freeze_r3gan:
            self.frozen_generator.eval()
            for param in self.frozen_generator.parameters():
                param.requires_grad = False

        # Wrap with DataParallel for multi-GPU; scatter/gather is handled
        # automatically while preserving gradient flow to the input latents.
        if device_ids is not None and len(device_ids) > 1:
            primary_device = torch.device(f'cuda:{device_ids[0]}')
            self.frozen_generator = self.frozen_generator.to(primary_device)
            self.frozen_generator = nn.DataParallel(self.frozen_generator, device_ids=device_ids)

        # --- Build MotionRNN ---
        self.use_content_motion = use_content_motion
        if use_content_motion:
            # MoCoGAN-style split: z_C is appearance (fixed), z_M is motion (predicted)
            motion_dim = z_dim - content_dim
            if content_dim < z_dim // 4:
                print(f"WARNING: content_dim={content_dim} is small relative to "
                      f"z_dim={z_dim}. Appearance capacity may be insufficient. "
                      f"Recommended: content_dim >= {z_dim // 4}")
            if motion_dim < z_dim // 4:
                print(f"WARNING: motion_dim={motion_dim} is small relative to "
                      f"z_dim={z_dim}. Motion expressiveness may be limited. "
                      f"Recommended: motion_dim >= {z_dim // 4}")
            self.motion_rnn = ContentMotionRNN(
                z_dim=z_dim,
                content_dim=content_dim,
                motion_dim=motion_dim,
                hidden_dim=motion_hidden_dim,
                num_layers=motion_num_layers,
                dropout=motion_dropout,
                use_noise_injection=True,
            )
        elif use_pca:
            self.motion_rnn = MotionRNNWithPCA(
                z_dim=z_dim,
                hidden_dim=motion_hidden_dim,
                num_layers=motion_num_layers,
                n_pca=n_pca,
                dropout=motion_dropout,
                residual_weight=residual_weight,
            )
        else:
            self.motion_rnn = MotionRNN(
                z_dim=z_dim,
                hidden_dim=motion_hidden_dim,
                num_layers=motion_num_layers,
                dropout=motion_dropout,
                residual_weight=residual_weight,
            )

        # --- Build SequenceEncoder (used for conditional generation) ---
        if use_encoder:
            self.encoder = SequenceEncoder(
                img_resolution=img_resolution,
                img_channels=3,
                w_dim=z_dim,
                hidden_dim=motion_hidden_dim,
                num_layers=motion_num_layers,
                base_channels=encoder_base_channels,
            )
        else:
            self.encoder = None

        # --- Build FateClassifier (Figure c: 'Fate Predict' block; Phi in the paper) ---
        # When enabled, the predicted class distribution replaces the ground-truth
        # label C that is passed to MotionRNN and the frozen generator.  This allows the model to
        # infer organoid fate from visual dynamics alone, without requiring a
        # manually assigned label at inference time.
        if use_fate_classifier:
            self.fate_classifier = FateClassifier(
                c_dim=c_dim,
                img_resolution=img_resolution,
                img_channels=3,
                base_channels=fate_classifier_base_channels,
                feat_dim=fate_classifier_feat_dim,
                hidden_dim=fate_classifier_hidden_dim,
                temperature=fate_classifier_temperature,
            )
        else:
            self.fate_classifier = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_frozen_generator(self, checkpoint_path: Union[str, Path]) -> nn.Module:
        """
        Load a pre-trained R3GAN Generator from a .pkl checkpoint, to be used as
        the frozen generator.

        The .pkl file is produced by the original r3gan/train.py and contains
        classes serialised under the module path ``training.networks.Generator``
        (relative to the r3gan/ subdirectory).  During unpickling we temporarily
        remap that path to ``models.temporal_organoid_generator.networks`` so that
        pickle can resolve the class; the remapping is reversed immediately after
        loading.

        Checkpoints produced before this module was renamed from
        ``models.r3gan`` to ``models.temporal_organoid_generator`` have that older
        module path baked into their pickle bytes (Python pickles classes by
        their defining ``__module__``, not by the path used to import them).
        We alias the old names too so those checkpoints keep loading unchanged.
        """
        checkpoint_path = Path(checkpoint_path)

        # Save current sys.modules state so we can restore it after loading.
        _alias_targets = {
            'training': 'models.temporal_organoid_generator',
            'training.networks': 'models.temporal_organoid_generator.networks',
            'R3GAN': 'models.temporal_organoid_generator',
            'R3GAN.Networks': 'models.temporal_organoid_generator.core',
            'R3GAN.FusedOperators': 'models.temporal_organoid_generator.operators',
            'R3GAN.Resamplers': 'models.temporal_organoid_generator.resamplers',
            # Backward compatibility with checkpoints pickled under the old
            # module name (pre-rename).
            'models.r3gan': 'models.temporal_organoid_generator',
            'models.r3gan.networks': 'models.temporal_organoid_generator.networks',
            'models.r3gan.core': 'models.temporal_organoid_generator.core',
            'models.r3gan.operators': 'models.temporal_organoid_generator.operators',
            'models.r3gan.resamplers': 'models.temporal_organoid_generator.resamplers',
        }
        _saved = {alias: sys.modules.get(alias) for alias in _alias_targets}

        try:
            # Map the pickle's class references to our local implementations.
            for alias, target in _alias_targets.items():
                sys.modules[alias] = importlib.import_module(target)

            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
        finally:
            # Always restore original sys.modules to avoid side effects.
            for name, module in _saved.items():
                if module is not None:
                    sys.modules[name] = module
                else:
                    sys.modules.pop(name, None)

        # Prefer the EMA snapshot (smoother samples) over the training snapshot.
        if 'G_ema' in data:
            generator = data['G_ema']
        elif 'G' in data:
            generator = data['G']
        else:
            raise ValueError(
                f"Cannot find Generator key ('G_ema' or 'G') in checkpoint: {checkpoint_path}"
            )

        return generator

    def _r3gan_forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass through the frozen generator.

        If DataParallel is active, scatter/gather is handled automatically.
        Note: we intentionally do NOT wrap this in torch.no_grad() even when
        the frozen generator is frozen, because we need gradients to flow back through it to
        update the Encoder and MotionRNN.  Its own parameters are protected
        via requires_grad=False.

        Args:
            z: [N, z_dim] — noise / latent vectors.
            c: [N, c_dim] — class-condition one-hot vectors.

        Returns:
            frames: [N, C, H, W] — generated images in [-1, 1].
        """
        return self.frozen_generator(z, c)

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        n_frames: int,
        return_z_seq: bool = False,
    ) -> torch.Tensor:
        """
        Unconditional video generation (noise → video).

        Uses MotionRNN to unfold the initial noise z into a temporal latent
        trajectory, then decodes each latent through R3GAN.

        Args:
            z: [B, z_dim] — initial noise sample.
            c: [B, c_dim] — class condition (one-hot).
            n_frames: Number of video frames to generate.
            return_z_seq: If True, also return the latent sequence and deltas.

        Returns:
            frames: [B, T, C, H, W] — generated video frames.
            z_seq:  [B, T, z_dim]   — latent sequence (only if return_z_seq).
            deltas: [B, T-1, z_dim] — per-step residuals (only if return_z_seq).
        """
        batch_size = z.shape[0]

        # Unfold initial latent into a temporal sequence via MotionRNN.
        z_seq, deltas = self.motion_rnn(z, n_frames, return_deltas=True)

        # Flatten batch × time for a single batched R3GAN forward pass.
        z_flat = z_seq.view(batch_size * n_frames, self.z_dim)
        c_flat = c.unsqueeze(1).expand(-1, n_frames, -1).reshape(batch_size * n_frames, self.c_dim)

        frames_flat = self._r3gan_forward(z_flat, c_flat)
        frames_flat = frames_flat.to(torch.float32)

        _, C, H, W = frames_flat.shape
        frames = frames_flat.view(batch_size, n_frames, C, H, W)

        if return_z_seq:
            return frames, z_seq, deltas

        return frames

    def generate_single_frame(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a single image from a given latent (bypasses MotionRNN).

        Useful for inspecting the static image generator in isolation.

        Args:
            z: [B, z_dim] — noise vector.
            c: [B, c_dim] — class condition.

        Returns:
            frame: [B, C, H, W] — generated image.
        """
        return self._r3gan_forward(z, c).to(torch.float32)

    def predict_from_frames(
        self,
        early_frames: torch.Tensor,
        c: Optional[torch.Tensor],
        total_frames: int,
        return_z_seq: bool = False,
        return_noise: bool = False,
        return_kl_params: bool = False,
        return_predicted_c: bool = False,
    ) -> torch.Tensor:
        """
        Conditional video generation: predict full development from early frames.

        This is the core inference method for V3Cell.  Given a short prefix of
        observed organoid frames (``early_frames``), the model:
          1. [Optional] FateClassifier (Phi) infers class C from early_frames.
             If use_fate_classifier=False (or fate_classifier is None), the
             caller-supplied ``c`` is used instead.
          2. Encodes each early frame to a latent z via the VAE-style SequenceEncoder.
          3. Uses the LSTM hidden state produced by the encoder to warm-start MotionRNN.
          4. Autoregressively generates future latents z_{k+1}, …, z_T.
          5. Decodes every latent (early + predicted) through the frozen generator
             generator to obtain the full video.

        Args:
            early_frames:     [B, T_in, C, H, W] — observed early frames in [-1, 1].
            c:                [B, c_dim] — ground-truth fate class (one-hot).
                              Ignored when use_fate_classifier=True; pass None in
                              that case if no ground-truth is available.
            total_frames:     Total number of frames in the output video (>= T_in).
            return_z_seq:     If True, return the full latent sequence and deltas.
            return_noise:     If True, return injected noise and its reconstruction
                              (needed for the mutual-information loss).
            return_kl_params: If True, return VAE μ and log σ² for the KL loss.
            return_predicted_c: If True, also return the predicted class distribution
                              from FateClassifier (useful for computing fate loss during
                              training, or for inspection at inference time).

        Returns (depending on flags):
            video:       [B, T_total, C, H, W] — full predicted video.
            predicted_c: [B, c_dim] — predicted class probs (if return_predicted_c).
            z_seq:       [B, T_total, z_dim]   — latent trajectory (optional).
            deltas:      per-step residuals     (optional).
            noise_in:    injected noise         (optional).
            noise_rec:   reconstructed noise    (optional).
            mu:          VAE mean               (optional).
            log_var:     VAE log-variance       (optional).
        """
        if self.encoder is None:
            raise RuntimeError(
                "No encoder attached. Initialise with use_encoder=True for "
                "conditional generation."
            )

        # Step 0 — Fate prediction (Figure c: 'Fate Predict' block; Phi in the paper).
        # If FateClassifier is attached, infer C from the visual dynamics of the
        # early frames.  This makes inference label-free: the model decides the
        # developmental fate on its own.
        predicted_c = None
        if self.fate_classifier is not None:
            predicted_c = self.fate_classifier(early_frames)  # [B, c_dim]
            c = predicted_c  # override the caller-supplied label
        elif c is None:
            raise ValueError(
                "c must be provided when use_fate_classifier=False."
            )

        B, T_input, C, H, W = early_frames.shape
        n_frames_to_generate = total_frames - T_input

        if n_frames_to_generate < 0:
            raise ValueError(
                f"total_frames ({total_frames}) must be >= number of input frames ({T_input})."
            )

        # Step 1 — Encode early frames to latent space (VAE-style).
        mu, log_var = None, None
        if return_kl_params:
            z_seq_input, (h_k, c_k), mu, log_var = self.encoder(
                early_frames, return_distribution=True
            )
        else:
            z_seq_input, (h_k, c_k) = self.encoder(early_frames)

        # Step 2 — Warm-start MotionRNN from the encoder hidden state and
        #          autoregressively predict future latents.
        noise_in = None
        noise_rec = None
        if n_frames_to_generate > 0:
            result = self.motion_rnn.forward_from_sequence(
                z_seq_input=z_seq_input,
                n_frames_to_generate=n_frames_to_generate,
                init_hidden=(h_k, c_k),
                return_deltas=True,
                return_noise=return_noise,
            )
            if return_noise:
                z_seq_full, deltas, noise_in, noise_rec = result
            else:
                z_seq_full, deltas = result
        else:
            # No future frames to predict; the early frames are the entire video.
            z_seq_full = z_seq_input
            deltas = None

        # Step 3 — Decode every latent through the frozen R3GAN.
        B, T_total, z_dim = z_seq_full.shape

        z_flat = z_seq_full.contiguous().view(B * T_total, z_dim)
        c_flat = c.unsqueeze(1).expand(-1, T_total, -1).reshape(B * T_total, self.c_dim)

        frames_flat = self._r3gan_forward(z_flat, c_flat).to(torch.float32)

        _, C_out, H_out, W_out = frames_flat.shape
        video = frames_flat.view(B, T_total, C_out, H_out, W_out)

        # Assemble return tuple based on requested extras.
        # predicted_c is prepended right after video when return_predicted_c=True,
        # so the caller can compute the fate classification loss against GT labels.
        if return_z_seq:
            if return_noise and return_kl_params:
                result = (video, z_seq_full, deltas, noise_in, noise_rec, mu, log_var)
            elif return_noise:
                result = (video, z_seq_full, deltas, noise_in, noise_rec)
            elif return_kl_params:
                result = (video, z_seq_full, deltas, mu, log_var)
            else:
                result = (video, z_seq_full, deltas)
        else:
            result = (video,)

        if return_predicted_c:
            # Insert predicted_c as the second element (after video)
            result = (result[0], predicted_c) + result[1:]

        return result if len(result) > 1 else result[0]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a frame sequence to the latent space (no LSTM warm-start).

        Args:
            frames: [B, T, C, H, W] — frame sequence.

        Returns:
            z_seq: [B, T, z_dim] — per-frame latent vectors.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder attached.")
        return self.encoder.encode_frames(frames)

    def sample_z(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample initial noise from the standard normal prior."""
        return torch.randn(batch_size, self.z_dim, device=device)

    def sample_c(
        self,
        batch_size: int,
        device: torch.device,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample class condition vectors.

        Args:
            batch_size: Batch size.
            device:     Target device.
            class_idx:  If given, all samples get this class; otherwise random.

        Returns:
            c: [B, c_dim] one-hot condition vectors.
        """
        if class_idx is not None:
            c = torch.zeros(batch_size, self.c_dim, device=device)
            c[:, class_idx] = 1
        else:
            indices = torch.randint(0, self.c_dim, (batch_size,), device=device)
            c = torch.zeros(batch_size, self.c_dim, device=device)
            c.scatter_(1, indices.unsqueeze(1), 1)
        return c

    def get_motion_params(self):
        """Return MotionRNN parameters (for building a separate optimizer group)."""
        return self.motion_rnn.parameters()

    def get_encoder_params(self):
        """Return Encoder parameters (for building a separate optimizer group)."""
        if self.encoder is not None:
            return self.encoder.parameters()
        return iter([])

    def get_fate_classifier_params(self):
        """Return FateClassifier parameters (for building a separate optimizer group)."""
        if self.fate_classifier is not None:
            return self.fate_classifier.parameters()
        return iter([])

    def get_trainable_params(self):
        """Return all trainable parameters: MotionRNN + Encoder + FateClassifier (frozen generator excluded)."""
        params = list(self.motion_rnn.parameters())
        if self.encoder is not None:
            params.extend(list(self.encoder.parameters()))
        if self.fate_classifier is not None:
            params.extend(list(self.fate_classifier.parameters()))
        return params

    def compute_motion_loss(self, deltas: torch.Tensor) -> torch.Tensor:
        """Delegate motion-smoothness regularisation to MotionRNN."""
        return self.motion_rnn.compute_motion_loss(deltas)

    def compute_reconstruction_loss(
        self,
        original_frames: torch.Tensor,
        reconstructed_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-pixel L1 reconstruction loss between original and predicted frames.

        Args:
            original_frames:     [B, T, C, H, W]
            reconstructed_frames:[B, T, C, H, W]

        Returns:
            loss: Scalar L1 loss.
        """
        return (original_frames - reconstructed_frames).abs().mean()

    def train(self, mode: bool = True):
        """Override train() to keep the frozen generator permanently in eval mode."""
        super().train(mode)
        if self.freeze_r3gan:
            # Handle the DataParallel wrapper if present.
            frozen_generator = self.frozen_generator.module if hasattr(self.frozen_generator, 'module') else self.frozen_generator
            frozen_generator.eval()
        return self

    def get_trainable_state_dict(self) -> dict:
        """
        Return a state dict containing only trainable weights
        (MotionRNN + Encoder + FateClassifier).

        Avoids storing the large frozen-generator weights in V3Cell checkpoints.
        """
        state = {'motion_rnn': self.motion_rnn.state_dict()}
        if self.encoder is not None:
            state['encoder'] = self.encoder.state_dict()
        if self.fate_classifier is not None:
            state['fate_classifier'] = self.fate_classifier.state_dict()
        return state

    def load_trainable_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Load a state dict produced by get_trainable_state_dict().

        Args:
            state_dict: Dictionary with keys 'motion_rnn', optionally 'encoder',
                        and optionally 'fate_classifier'.
            strict:     Whether to require an exact key match.
        """
        if 'motion_rnn' in state_dict:
            self.motion_rnn.load_state_dict(state_dict['motion_rnn'], strict=strict)
        else:
            # Legacy format: the dict is the MotionRNN state dict directly.
            self.motion_rnn.load_state_dict(state_dict, strict=strict)

        if 'encoder' in state_dict and self.encoder is not None:
            self.encoder.load_state_dict(state_dict['encoder'], strict=strict)

        if 'fate_classifier' in state_dict and self.fate_classifier is not None:
            self.fate_classifier.load_state_dict(
                state_dict['fate_classifier'], strict=strict
            )
