# Third-Party Notices

This repository contains, in addition to the original work of the V3Cell authors
(licensed under CC BY-NC 4.0, see [`LICENSE`](LICENSE)), third-party components that
are governed by their **own** licenses. Those licenses are retained in the source file
headers and take precedence for the corresponding files. Several of them are
**non-commercial / research-only**, which means the repository as a whole may only be
used for non-commercial research purposes.

---

## 1. NVIDIA StyleGAN3 (and derived code)

**Affected paths (non-exhaustive):**

- `dnnlib/`
- `torch_utils/`
- `r3gan/` — including `r3gan/train.py`, `r3gan/legacy.py`, `r3gan/gen_images.py`,
  `r3gan/calc_metrics.py`, `r3gan/dataset_tool.py`, `r3gan/training/`, `r3gan/metrics/`
- `models/temporal_organoid_generator/networks.py`

**Copyright:** © 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

**License:** NVIDIA Source Code License (non-commercial, research-only). The original
header, reproduced in each affected file, states that any use, reproduction, disclosure,
or distribution without an express license agreement from NVIDIA CORPORATION is
prohibited.

**Upstream:** https://github.com/NVlabs/stylegan3 —
license: https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt

> These files retain NVIDIA's original copyright and license headers and **must not be
> relicensed**. The CC BY-NC 4.0 license of this repository does **not** apply to them.

---

## 2. R3GAN

The Stage 1 generator (`r3gan/`) is derived from **R3GAN** (Huang et al., *"The GAN is
dead; long live the GAN! A Modern GAN Baseline"*, NeurIPS 2024), which itself builds on
NVIDIA StyleGAN3. The StyleGAN3-lineage files listed in Section 1 apply here as well.

**Upstream:** https://github.com/brownvc/R3GAN

---

## Summary

| Component | Paths | License | Commercial use |
|---|---|---|---|
| V3Cell original work | everything else | CC BY-NC 4.0 | Not permitted |
| NVIDIA StyleGAN3 / R3GAN | `dnnlib/`, `torch_utils/`, `r3gan/`, `models/temporal_organoid_generator/networks.py` | NVIDIA Source Code License | Not permitted |

If you intend to use any part of this repository beyond non-commercial academic
research, you must independently obtain the appropriate rights from the respective
copyright holders (including NVIDIA CORPORATION).
