# StyleGAN2 vs R3GAN Interface Comparison

## 1. Generator Interface Comparison

### StyleGAN2 Generator (MoCoGAN-HD)

```python
class Generator(nn.Module):
    def __init__(self,
                 size,           # output image size (256, 512, 1024...)
                 style_dim,      # latent dimension (512)
                 n_mlp,          # mapping network depth (8)
                 ...):

    def forward(self,
                styles,          # input: list of [B, style_dim] noise vectors
                n_frame,         # number of video frames
                ...):
        # 1. mapping network: z -> w
        styles = [self.style(s) for s in styles]

        # 2. RNN generates temporal w sequence
        out, rand_in, rand_rec = self.modelR(styles[0], n_frame)

        # 3. synthesis network: w -> image
        # ... generate frames
        return image  # [B*n_frame, 3, H, W]
```

**Key properties:**
- Uses **W-space** (latent mapped through the mapping network)
- Mapping network: 8-layer MLP mapping z → w
- Supports style mixing (multiple w vectors injected at different layers)
- Input dimension: `z_dim = 512`
- Output: `[B*n_frame, 3, H, W]`

---

### R3GAN Generator

```python
class Generator(nn.Module):
    def __init__(self,
                 NoiseDimension,              # latent dimension (64)
                 WidthPerStage,               # channels per stage
                 CardinalityPerStage,         # group count per stage
                 BlocksPerStage,              # residual blocks per stage
                 ExpansionFactor,             # expansion factor
                 ConditionDimension=None,     # condition dimension (number of classes)
                 ConditionEmbeddingDimension=0,
                 ...):

    def forward(self, x, y=None):
        # x: [B, z_dim] noise
        # y: [B, c_dim] condition (one-hot class label)

        # 1. Concatenate noise and condition embedding
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1)

        # 2. Pass through residual network
        for Layer in self.MainLayers:
            x = Layer(x)

        return self.AggregationLayer(x)  # [B, 3, H, W]
```

**Key properties:**
- Uses **Z-space** (noise used directly, no mapping network)
- **No mapping network**
- Native support for conditional generation (class one-hot)
- Input dimension: `z_dim = 64` (smaller)
- Output: `[B, 3, H, W]`

---

## 2. Core Differences

| Feature | StyleGAN2 | R3GAN |
|---------|-----------|-------|
| **Latent space** | W-space (mapped) | Z-space (direct) |
| **z_dim** | 512 | 64 |
| **Mapping network** | Yes (8-layer MLP) | No |
| **Conditional input** | None / limited | Native (one-hot) |
| **Style injection** | AdaIN / Modulation | Concat + residual |
| **Architecture** | StyleBlock | ResBlock |
| **Main advantage** | High quality, controllable | Stable training, simple structure |

---

## 3. Key Components in MoCoGAN-HD

### RNN Module (motion generator)

```python
class RNNModule(nn.Module):
    def __init__(self,
                 z_dim=512,      # matches StyleGAN2 dimension
                 h_dim=384,      # LSTM hidden dimension
                 n_pca=384,      # number of PCA components
                 w_residual=0.2, # residual weight
                 ...):

    def forward(self, z, n_frame):
        # z: [B, 512] initial w vector
        # n_frame: number of frames

        out = [z]
        for i in range(n_frame - 1):
            # LSTM generates motion delta
            h_, c_ = self.cell(e_, (h[-1], c[-1]))
            mul = torch.tanh(torch.matmul(h_, self.w) + self.b)

            # Add motion in PCA space
            out_ = out[-1] + self.w_residual * torch.matmul(mul, pca_mul)
            out.append(out_)

        return out  # [B*n_frame, 512]
```

**Key points:**
- Input: W-space vector (512-dim)
- Uses PCA decomposition to find motion directions in W-space
- LSTM predicts motion deltas
- Residual motion: `w_t = w_{t-1} + delta`

---

## 4. Adaptation Strategies

### Strategy A: Direct replacement (simple)

```python
# Expand R3GAN z_dim from 64 to 512, or let RNN output 64-dim vectors

class MotionRNN(nn.Module):
    def __init__(self, z_dim=64, h_dim=128):  # adapted for R3GAN
        self.cell = nn.LSTMCell(z_dim, h_dim)
        self.fc = nn.Linear(h_dim, z_dim)

    def forward(self, z, n_frame, c=None):
        # z: [B, 64]
        # c: [B, c_dim] condition (unchanged)

        out = [z]
        h, c_state = self.cell(z)

        for i in range(n_frame - 1):
            delta = self.fc(h)
            z_new = out[-1] + 0.1 * delta  # residual motion
            out.append(z_new)
            h, c_state = self.cell(z_new, (h, c_state))

        return torch.stack(out, dim=1)  # [B, n_frame, 64]
```

**Pros:** Simple to implement; preserves R3GAN conditional generation capability
**Cons:** No PCA constraint; motion may be unnatural

---

### Strategy B: Full port (recommended)

```python
class R3GANVideoGenerator(nn.Module):
    def __init__(self, r3gan_ckpt, z_dim=64, h_dim=128, n_pca=64):
        # 1. Load pre-trained R3GAN
        self.G = load_r3gan(r3gan_ckpt)
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad = False

        # 2. Learn PCA in Z-space
        self.pca_comp = nn.Parameter(...)  # learned from data

        # 3. Motion RNN
        self.motion_rnn = MotionRNN(z_dim, h_dim, n_pca)

    def forward(self, z, c, n_frame):
        # z: [B, 64] initial noise
        # c: [B, c_dim] condition
        # n_frame: number of frames

        # Generate motion sequence
        z_seq = self.motion_rnn(z, n_frame)  # [B, n_frame, 64]

        # Generate each frame
        B, T, _ = z_seq.shape
        z_flat = z_seq.view(B*T, -1)
        c_flat = c.unsqueeze(1).expand(-1, T, -1).reshape(B*T, -1)

        frames = self.G(z_flat, c_flat)  # [B*T, 3, H, W]

        return frames.view(B, T, 3, H, W)
```

---

## 5. Key Problems to Solve

### Problem 1: Z-space vs W-space

**StyleGAN2 advantage:**
- W-space is more "disentangled"; different dimensions control different attributes
- PCA can find meaningful motion directions

**R3GAN situation:**
- Z-space has no mapping network
- Need to verify whether Z-space also has disentangled motion directions
- May require learning a lightweight transform

### Problem 2: Dimension mismatch

| Model | Latent dimension |
|-------|-----------------|
| StyleGAN2 | 512 |
| R3GAN | 64 |

**Solutions:**
- Keep 64-dim and adjust RNN structure accordingly
- Or expand R3GAN's z_dim (requires retraining)

### Problem 3: Conditional generation

**R3GAN advantage:** native conditional generation support
**Application:** generate organoid videos conditioned on a specific fate class

### Problem 4: PCA motion directions

Learn motion directions from R3GAN's Z-space:
1. Collect a large set of z vectors and their corresponding images
2. Compute z differences between adjacent frames
3. Apply PCA to the differences to find principal motion directions

---

## 6. Recommended Implementation Steps

1. **Phase 1: Validate feasibility**
   - Simple RNN + R3GAN
   - No PCA — predict z deltas directly
   - Verify that the model can generate temporally coherent frames

2. **Phase 2: Learn motion space**
   - Collect organoid video data
   - Learn PCA components in Z-space
   - Build a motion dictionary

3. **Phase 3: Full training**
   - Add 3D discriminator
   - Add 2D discriminator
   - Adversarial training

4. **Phase 4: Evaluation and refinement**
   - FVD (Fréchet Video Distance)
   - Visual quality assessment
   - Temporal coherence assessment
