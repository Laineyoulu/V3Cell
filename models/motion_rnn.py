# -*- coding: utf-8 -*-
"""
MotionRNN: Temporal latent-trajectory generator for V3Cell.

Generates a temporally coherent sequence of latent vectors from an initial
noise sample by running an LSTM that predicts per-step residual updates.

Variants:
  - MotionRNN:          Standard version with optional per-step noise injection.
  - ContentMotionRNN:   MoCoGAN-style split: z = [z_C (fixed), z_M (dynamic)].
  - MotionRNNWithPCA:   Generates motion in a learned PCA subspace of z.

Noise injection + mutual-information loss (following MoCoGAN-HD):
  At each step a fresh noise vector eps_t is injected into the LSTM input.
  A small reconstruction network is trained to predict eps_t from the LSTM
  hidden state, creating a mutual-information pressure that prevents the LSTM
  from ignoring the stochastic input (motion mode collapse).
"""

import torch
import torch.nn as nn
from torch.nn import init


class MotionRNN(nn.Module):
    """
    Standard Motion RNN with optional per-step noise injection.

    Unfolds an initial latent z_0 into a trajectory [z_0, z_1, …, z_T] by
    running an LSTM and applying residual updates:

        z_t = z_{t-1} + residual_weight * delta_t

    where delta_t = Decoder(LSTM(z_{t-1} ⊕ eps_t)).

    The noise eps_t is drawn i.i.d. from N(0, I) at every step, providing
    stochastic diversity in the generated trajectories.  A small reconstruction
    head is optionally trained to predict eps_t back from the LSTM state,
    providing a mutual-information loss that prevents motion mode collapse.
    """

    def __init__(
        self,
        z_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual_weight: float = 0.1,
        use_noise_injection: bool = True,
    ):
        """
        Args:
            z_dim:             Latent vector dimension.
            hidden_dim:        LSTM hidden state size.
            num_layers:        Number of stacked LSTM layers.
            dropout:           Dropout probability (applied between LSTM layers).
            residual_weight:   Scale factor for the residual update Δz.
            use_noise_injection: Inject fresh noise at each step and attach a
                                 reconstruction head for the MI loss.
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual_weight = residual_weight
        self.use_noise_injection = use_noise_injection

        # Encoder: maps initial z to the LSTM initial hidden state.
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # LSTM core.  When noise injection is enabled the input is [z; eps],
        # doubling the input size compared to z alone.
        lstm_input_size = z_dim * 2 if use_noise_injection else z_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder: maps LSTM output to a residual Δz in z-space.
        # Tanh constrains the magnitude of each step.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, z_dim),
            nn.Tanh(),
        )

        # Noise reconstruction head: predicts eps_t from (h_t, c_t).
        # Used to compute the mutual-information loss.
        if use_noise_injection:
            self.noise_reconstructor = nn.Sequential(
                nn.Linear(hidden_dim * 2, z_dim),
                nn.ReLU(inplace=True),
                nn.Linear(z_dim, z_dim),
            )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialisation for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def _init_hidden(self, z: torch.Tensor) -> tuple:
        """
        Derive the initial LSTM hidden state from the initial latent z.

        Args:
            z: [B, z_dim] — initial latent vectors.

        Returns:
            (h_k, c_k): LSTM hidden and cell states of shape [num_layers, B, hidden_dim].
        """
        encoded = self.encoder(z)  # [B, hidden_dim]
        h_k = encoded.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_k = torch.zeros_like(h_k)
        return h_k, c_k

    def forward(
        self,
        z: torch.Tensor,
        n_frames: int,
        return_deltas: bool = False,
        return_noise: bool = False,
        init_hidden: tuple = None,
    ):
        """
        Generate a temporal latent sequence of length n_frames.

        Args:
            z:            [B, z_dim] — initial latent (first frame).
            n_frames:     Total number of frames (sequence length).
            return_deltas: If True, also return the per-step residual deltas.
            return_noise:  If True, also return injected noise and its
                           reconstruction (for the MI loss).
            init_hidden:   Optional (h_k, c_k) to override default initialisation.

        Returns:
            z_seq:     [B, n_frames, z_dim]
            deltas:    [B, n_frames-1, z_dim]  (if return_deltas)
            noise_in:  [B, n_frames-1, z_dim]  (if return_noise)
            noise_rec: [B, n_frames-1, z_dim]  (if return_noise)
        """
        batch_size = z.shape[0]
        device = z.device

        h, c = init_hidden if init_hidden is not None else self._init_hidden(z)

        z_sequence = [z]
        deltas = []
        noise_inputs = []
        noise_reconstructed = []

        z_current = z

        for t in range(n_frames - 1):
            if self.use_noise_injection:
                eps = torch.randn(batch_size, self.z_dim, device=device)
                noise_inputs.append(eps)
                lstm_input = torch.cat([z_current, eps], dim=-1).unsqueeze(1)  # [B, 1, z_dim*2]
            else:
                lstm_input = z_current.unsqueeze(1)  # [B, 1, z_dim]

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

            # Reconstruct the injected noise from the LSTM state (MI loss support).
            if self.use_noise_injection and return_noise:
                hc = torch.cat([h[-1], c[-1]], dim=-1)  # [B, hidden_dim*2]
                noise_reconstructed.append(self.noise_reconstructor(hc))

            delta = self.decoder(lstm_out.squeeze(1))  # [B, z_dim]
            deltas.append(delta)

            z_current = z_current + self.residual_weight * delta
            z_sequence.append(z_current)

        z_seq = torch.stack(z_sequence, dim=1)  # [B, n_frames, z_dim]

        result = [z_seq]

        if return_deltas:
            result.append(torch.stack(deltas, dim=1))  # [B, n_frames-1, z_dim]
        else:
            result.append(None)

        if return_noise and self.use_noise_injection:
            result.append(torch.stack(noise_inputs, dim=1))
            result.append(torch.stack(noise_reconstructed, dim=1))
        elif return_noise:
            result.extend([None, None])

        if return_deltas or return_noise:
            return tuple(result)
        return z_seq

    def forward_from_sequence(
        self,
        z_seq_input: torch.Tensor,
        n_frames_to_generate: int,
        init_hidden: tuple = None,
        return_deltas: bool = False,
        return_noise: bool = False,
    ):
        """
        Continue a latent trajectory given a known prefix sequence.

        This is the core conditional-generation method used by V3CellGenerator:
          - Input:  observed latents [z_0, z_1, …, z_k] from early frames.
          - Output: full trajectory [z_0, …, z_k, z_{k+1}, …, z_{k+n}].

        The LSTM is first "warmed up" by feeding the known latents one by one
        (with zero noise to avoid injecting uncertainty into the observed region),
        then prediction begins from z_k.

        Args:
            z_seq_input:         [B, T_in, z_dim] — latents of the early frames.
            n_frames_to_generate: Number of future frames to predict.
            init_hidden:         Optional (h_k, c_k) override.
            return_deltas:       If True, return predicted per-step residuals.
            return_noise:        If True, return noise and its reconstruction.

        Returns:
            z_seq_full: [B, T_in + n_frames_to_generate, z_dim]
            deltas:     [B, n_frames_to_generate, z_dim]  (optional)
            noise_in:   [B, n_frames_to_generate, z_dim]  (optional)
            noise_rec:  [B, n_frames_to_generate, z_dim]  (optional)
        """
        B, T_input, z_dim = z_seq_input.shape
        device = z_seq_input.device

        h, c = (
            init_hidden if init_hidden is not None
            else self._init_hidden(z_seq_input[:, 0])
        )

        # Warm-up: feed the known latents to build up the LSTM hidden state.
        # Zero noise is used here because these frames are already observed.
        for t in range(T_input):
            z_t = z_seq_input[:, t]
            if self.use_noise_injection:
                zero_noise = torch.zeros(B, self.z_dim, device=device)
                lstm_input = torch.cat([z_t, zero_noise], dim=-1).unsqueeze(1)
            else:
                lstm_input = z_t.unsqueeze(1)
            _, (h, c) = self.lstm(lstm_input, (h, c))

        # Autoregressive prediction from the last known latent.
        z_current = z_seq_input[:, -1]
        z_predicted = []
        deltas = []
        noise_inputs = []
        noise_reconstructed = []

        for t in range(n_frames_to_generate):
            if self.use_noise_injection:
                eps = torch.randn(B, self.z_dim, device=device)
                noise_inputs.append(eps)
                lstm_input = torch.cat([z_current, eps], dim=-1).unsqueeze(1)
            else:
                lstm_input = z_current.unsqueeze(1)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

            if self.use_noise_injection and return_noise:
                hc = torch.cat([h[-1], c[-1]], dim=-1)
                noise_reconstructed.append(self.noise_reconstructor(hc))

            delta = self.decoder(lstm_out.squeeze(1))
            deltas.append(delta)

            z_current = z_current + self.residual_weight * delta
            z_predicted.append(z_current)

        # Concatenate observed prefix with predicted suffix.
        if z_predicted:
            z_predicted_t = torch.stack(z_predicted, dim=1)  # [B, n_gen, z_dim]
            z_seq_full = torch.cat([z_seq_input, z_predicted_t], dim=1)
        else:
            z_seq_full = z_seq_input

        result = [z_seq_full]

        if return_deltas:
            result.append(
                torch.stack(deltas, dim=1) if deltas
                else torch.zeros(B, 0, z_dim, device=device)
            )
        else:
            result.append(None)

        if return_noise and self.use_noise_injection:
            result.append(
                torch.stack(noise_inputs, dim=1) if noise_inputs
                else torch.zeros(B, 0, z_dim, device=device)
            )
            result.append(
                torch.stack(noise_reconstructed, dim=1) if noise_reconstructed
                else torch.zeros(B, 0, z_dim, device=device)
            )
        elif return_noise:
            result.extend([None, None])

        if return_deltas or return_noise:
            return tuple(result)
        return z_seq_full

    def compute_motion_loss(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Motion-smoothness regularisation loss.

        Penalises large residuals (encourages small per-step changes) and
        penalises abrupt changes between consecutive residuals (encourages
        smooth acceleration).

        Args:
            deltas: [B, T-1, z_dim] — per-step latent residuals.

        Returns:
            loss: Scalar motion regularisation loss.
        """
        magnitude_loss = deltas.pow(2).mean()

        if deltas.shape[1] > 1:
            delta_diff = deltas[:, 1:] - deltas[:, :-1]
            smoothness_loss = delta_diff.pow(2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=deltas.device)

        return magnitude_loss + 0.5 * smoothness_loss


class ContentMotionRNN(nn.Module):
    """
    MoCoGAN-style Content-Motion separated MotionRNN.

    Key design principle from MoCoGAN:
      - z_C (Content Code): fixed throughout the video; encodes identity/appearance.
      - z_M (Motion Code):  predicted by the LSTM at each time step.
      - Full latent:        z = [z_C, z_M]  (simple concatenation).

    For conditional generation:
      - z_C is extracted from the first observed frame and held constant.
      - z_M is unrolled autoregressively by the LSTM.

    This separation ensures appearance consistency while allowing free motion.
    """

    def __init__(
        self,
        z_dim: int = 64,
        content_dim: int = 32,
        motion_dim: int = 32,
        hidden_dim: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_noise_injection: bool = True,
    ):
        """
        Args:
            z_dim:        Total latent dimension (must equal content_dim + motion_dim).
            content_dim:  Dimension of the fixed content code z_C.
            motion_dim:   Dimension of the time-varying motion code z_M.
            hidden_dim:   LSTM hidden state size.
            num_layers:   Number of stacked LSTM layers.
            dropout:      Dropout probability between LSTM layers.
            use_noise_injection: Inject noise into z_M at each step (for MI loss).
        """
        super().__init__()

        self.z_dim = z_dim
        self.content_dim = content_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_noise_injection = use_noise_injection

        assert content_dim + motion_dim == z_dim, (
            f"content_dim ({content_dim}) + motion_dim ({motion_dim}) "
            f"must equal z_dim ({z_dim})."
        )

        # Encoder: maps initial z_M to the LSTM initial hidden state.
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # LSTM operates only on the motion code z_M.
        lstm_input_size = motion_dim * 2 if use_noise_injection else motion_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder: outputs residual Δz_M in the motion subspace.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, motion_dim),
            nn.Tanh(),
        )

        # Noise reconstruction head for the mutual-information loss.
        if use_noise_injection:
            self.noise_reconstructor = nn.Sequential(
                nn.Linear(hidden_dim * 2, motion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(motion_dim, motion_dim),
            )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def split_content_motion(self, z: torch.Tensor) -> tuple:
        """
        Split the full latent z into (z_C, z_M).

        Args:
            z: [B, z_dim] or [B, T, z_dim]

        Returns:
            z_C: [..., content_dim]
            z_M: [..., motion_dim]
        """
        if z.dim() == 2:
            return z[:, :self.content_dim], z[:, self.content_dim:]
        return z[:, :, :self.content_dim], z[:, :, self.content_dim:]

    def combine_content_motion(self, z_C: torch.Tensor, z_M: torch.Tensor) -> torch.Tensor:
        """
        Concatenate content code z_C and motion code z_M to form z.

        Args:
            z_C: [..., content_dim]
            z_M: [..., motion_dim]

        Returns:
            z: [..., z_dim]
        """
        return torch.cat([z_C, z_M], dim=-1)

    def _init_hidden(self, z_M: torch.Tensor) -> tuple:
        """Derive LSTM initial state from the initial motion code z_M."""
        encoded = self.encoder(z_M)
        h_k = encoded.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_k = torch.zeros_like(h_k)
        return h_k, c_k

    def forward(
        self,
        z: torch.Tensor,
        n_frames: int,
        return_deltas: bool = False,
        return_noise: bool = False,
        init_hidden: tuple = None,
    ):
        """
        Unconditional latent trajectory generation with content-motion separation.

        z is split into a fixed content code z_C (held constant for every frame)
        and an initial motion code z_M, which is then unrolled autoregressively
        by the LSTM exactly as in ``MotionRNN.forward``. Mirrors the signature of
        ``MotionRNN.forward`` so callers (e.g. ``V3CellGenerator.forward``) do not
        need to know which RNN variant they hold.

        Args:
            z:            [B, z_dim] — initial latent (first frame).
            n_frames:     Total number of frames (sequence length).
            return_deltas: If True, also return the per-step residual deltas (in
                           the motion subspace, [B, n_frames-1, motion_dim]).
            return_noise:  If True, also return injected noise and its
                           reconstruction (for the MI loss).
            init_hidden:   Optional (h_k, c_k) to override default initialisation.

        Returns:
            z_seq:     [B, n_frames, z_dim]
            deltas:    [B, n_frames-1, motion_dim]  (if return_deltas)
            noise_in:  [B, n_frames-1, motion_dim]  (if return_noise)
            noise_rec: [B, n_frames-1, motion_dim]  (if return_noise)
        """
        batch_size = z.shape[0]
        device = z.device

        z_C_fixed, z_M_current = self.split_content_motion(z)

        h, c = init_hidden if init_hidden is not None else self._init_hidden(z_M_current)

        z_M_sequence = [z_M_current]
        deltas = []
        noise_inputs = []
        noise_reconstructed = []

        for t in range(n_frames - 1):
            if self.use_noise_injection:
                eps = torch.randn(batch_size, self.motion_dim, device=device)
                noise_inputs.append(eps)
                lstm_input = torch.cat([z_M_current, eps], dim=-1).unsqueeze(1)
            else:
                lstm_input = z_M_current.unsqueeze(1)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

            if self.use_noise_injection and return_noise:
                hc = torch.cat([h[-1], c[-1]], dim=-1)
                noise_reconstructed.append(self.noise_reconstructor(hc))

            delta = self.decoder(lstm_out.squeeze(1))
            deltas.append(delta)

            z_M_current = z_M_current + 0.2 * delta  # residual_weight = 0.2
            z_M_sequence.append(z_M_current)

        z_M_seq = torch.stack(z_M_sequence, dim=1)  # [B, n_frames, motion_dim]
        z_C_expanded = z_C_fixed.unsqueeze(1).expand(-1, n_frames, -1)
        z_seq = self.combine_content_motion(z_C_expanded, z_M_seq)

        result = [z_seq]

        if return_deltas:
            result.append(torch.stack(deltas, dim=1))
        else:
            result.append(None)

        if return_noise and self.use_noise_injection:
            result.append(torch.stack(noise_inputs, dim=1))
            result.append(torch.stack(noise_reconstructed, dim=1))
        elif return_noise:
            result.extend([None, None])

        if return_deltas or return_noise:
            return tuple(result)
        return z_seq

    def forward_from_sequence(
        self,
        z_seq_input: torch.Tensor,
        n_frames_to_generate: int,
        init_hidden: tuple = None,
        return_deltas: bool = False,
        return_noise: bool = False,
    ):
        """
        Conditional latent prediction with content-motion separation.

        z_C is taken from the first input frame and broadcast to all generated
        frames; z_M is predicted autoregressively by the LSTM.

        Args:
            z_seq_input:          [B, T_in, z_dim] — observed latent prefix.
            n_frames_to_generate: Number of future frames to predict.
            init_hidden:          Optional (h_k, c_k) override.
            return_deltas:        If True, return Δz_M for each generated step.
            return_noise:         If True, return eps and its reconstruction.

        Returns:
            z_seq_full: [B, T_in + n_gen, z_dim]
            (plus optional deltas, noise_in, noise_rec)
        """
        B, T_input, _ = z_seq_input.shape
        device = z_seq_input.device

        z_C_seq, z_M_seq = self.split_content_motion(z_seq_input)

        # Content code is fixed to the first frame's appearance.
        z_C_fixed = z_C_seq[:, 0]  # [B, content_dim]

        h, c = (
            init_hidden if init_hidden is not None
            else self._init_hidden(z_M_seq[:, 0])
        )

        # Warm-up LSTM on the observed motion codes.
        for t in range(T_input):
            z_M_t = z_M_seq[:, t]
            if self.use_noise_injection:
                zero_noise = torch.zeros(B, self.motion_dim, device=device)
                lstm_input = torch.cat([z_M_t, zero_noise], dim=-1).unsqueeze(1)
            else:
                lstm_input = z_M_t.unsqueeze(1)
            _, (h, c) = self.lstm(lstm_input, (h, c))

        z_M_current = z_M_seq[:, -1]
        z_M_predicted = []
        deltas = []
        noise_inputs = []
        noise_reconstructed = []

        for t in range(n_frames_to_generate):
            if self.use_noise_injection:
                eps = torch.randn(B, self.motion_dim, device=device)
                noise_inputs.append(eps)
                lstm_input = torch.cat([z_M_current, eps], dim=-1).unsqueeze(1)
            else:
                lstm_input = z_M_current.unsqueeze(1)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

            if self.use_noise_injection and return_noise:
                hc = torch.cat([h[-1], c[-1]], dim=-1)
                noise_reconstructed.append(self.noise_reconstructor(hc))

            delta = self.decoder(lstm_out.squeeze(1))
            deltas.append(delta)

            z_M_current = z_M_current + 0.2 * delta  # residual_weight = 0.2
            z_M_predicted.append(z_M_current)

        # Assemble full trajectory.
        if z_M_predicted:
            z_M_pred_t = torch.stack(z_M_predicted, dim=1)  # [B, n_gen, motion_dim]
            z_C_expanded = z_C_fixed.unsqueeze(1).expand(-1, n_frames_to_generate, -1)
            z_predicted = self.combine_content_motion(z_C_expanded, z_M_pred_t)
            z_seq_full = torch.cat([z_seq_input, z_predicted], dim=1)
        else:
            z_seq_full = z_seq_input

        result = [z_seq_full]

        if return_deltas:
            result.append(
                torch.stack(deltas, dim=1) if deltas
                else torch.zeros(B, 0, self.motion_dim, device=device)
            )
        else:
            result.append(None)

        if return_noise and self.use_noise_injection:
            result.append(
                torch.stack(noise_inputs, dim=1) if noise_inputs
                else torch.zeros(B, 0, self.motion_dim, device=device)
            )
            result.append(
                torch.stack(noise_reconstructed, dim=1) if noise_reconstructed
                else torch.zeros(B, 0, self.motion_dim, device=device)
            )
        elif return_noise:
            result.extend([None, None])

        if return_deltas or return_noise:
            return tuple(result)
        return z_seq_full

    def compute_motion_loss(self, deltas: torch.Tensor) -> torch.Tensor:
        """Motion-smoothness regularisation (same as MotionRNN)."""
        if deltas is None or deltas.numel() == 0:
            return torch.tensor(0.0)

        magnitude_loss = deltas.pow(2).mean()

        if deltas.shape[1] > 1:
            delta_diff = deltas[:, 1:] - deltas[:, :-1]
            smoothness_loss = delta_diff.pow(2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=deltas.device)

        return magnitude_loss + 0.5 * smoothness_loss


class MotionRNNWithPCA(MotionRNN):
    """
    MotionRNN variant that generates motion in a learned PCA subspace.

    Instead of producing a residual Δz directly, the LSTM outputs a coefficient
    vector over n_pca learnable principal components.  This constrains motion to
    lie in a low-dimensional manifold, which can improve visual coherence.

    Can be initialised from real latent-delta statistics via init_pca_from_data().
    """

    def __init__(
        self,
        z_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        n_pca: int = 32,
        dropout: float = 0.1,
        residual_weight: float = 0.1,
    ):
        super().__init__(z_dim, hidden_dim, num_layers, dropout, residual_weight)

        self.n_pca = n_pca

        # Learnable PCA components (can be pre-initialised from data).
        self.pca_components = nn.Parameter(torch.randn(n_pca, z_dim) * 0.01)
        self.pca_std = nn.Parameter(torch.ones(n_pca))

        # Override decoder to output PCA coefficients instead of a raw Δz.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, n_pca),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
        n_frames: int,
        return_deltas: bool = False,
    ):
        """Generate a latent trajectory using PCA-constrained motion."""
        batch_size = z.shape[0]
        device = z.device

        h, c = self._init_hidden(z)

        z_sequence = [z]
        deltas = []
        z_current = z

        # Scaled PCA basis: component i has magnitude proportional to pca_std[i].
        pca_transform = self.pca_components * self.pca_std.unsqueeze(1)  # [n_pca, z_dim]

        for t in range(n_frames - 1):
            lstm_input = z_current.unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

            # LSTM output → PCA coefficients → Δz in z-space.
            pca_coef = self.decoder(lstm_out.squeeze(1))    # [B, n_pca]
            delta = torch.matmul(pca_coef, pca_transform)   # [B, z_dim]
            deltas.append(delta)

            z_current = z_current + self.residual_weight * delta
            z_sequence.append(z_current)

        z_seq = torch.stack(z_sequence, dim=1)

        if return_deltas:
            return z_seq, torch.stack(deltas, dim=1)
        return z_seq

    def init_pca_from_data(self, z_deltas: torch.Tensor):
        """
        Initialise PCA components from a collection of observed latent deltas.

        Args:
            z_deltas: [N, z_dim] — sample of Δz vectors collected from real data.
        """
        from sklearn.decomposition import PCA

        z_np = z_deltas.detach().cpu().numpy()
        pca = PCA(n_components=self.n_pca)
        pca.fit(z_np)

        with torch.no_grad():
            self.pca_components.copy_(torch.from_numpy(pca.components_))
            self.pca_std.copy_(
                torch.from_numpy(pca.singular_values_ / len(z_np) ** 0.5)
            )
