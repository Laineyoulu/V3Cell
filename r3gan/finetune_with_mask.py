#!/usr/bin/env python3
"""
R3GAN fine-tuning script with foreground mask weighting.

Usage:
    python finetune_with_mask.py \\
        --resume=training-runs/00000-lung-64x64-gpus1-batch32-lung/network-snapshot-000001000.pkl \\
        --data=datasets/lung-64x64.zip \\
        --mask-path=datasets/lung-masks-64x64.zip \\
        --foreground-weight=5.0 \\
        --kimg=500 \\
        --outdir=finetuned-runs

Features:
    - Loads pre-trained model weights
    - Uses Masked R1 Regularisation
    - Fully self-contained — does not modify the original training code
    - Data augmentation disabled during fine-tuning (avoids mask/image misalignment)
    - Output format identical to the original training run
"""

import os
import sys
from pathlib import Path

# Prioritise r3gan/ so that `from training import ...` resolves to r3gan/training/
_R3GAN_DIR = str(Path(__file__).parent)
if _R3GAN_DIR not in sys.path:
    sys.path.insert(0, _R3GAN_DIR)

# Project root comes second so that torch_utils and dnnlib can be imported directly
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)

import time
import copy
import json
import pickle
import psutil
import click
import torch
import numpy as np
import PIL.Image
import dnnlib
import legacy
from torch_utils import training_stats
from torch_utils import misc
from training.dataset_with_mask import ImageFolderDatasetWithMask
from training.loss_with_mask import R3GANLossWithMask


def save_image_grid(img, fname, drange, grid_size):
    """Save a grid of images (same format as the original training code)."""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


@click.command()
# Required arguments
@click.option('--resume', help='Pre-trained model path (.pkl)', required=True, metavar='PKL')
@click.option('--data', help='Dataset path (zip or directory)', required=True, metavar='PATH')
@click.option('--outdir', help='Output directory', required=True, metavar='DIR')

# Mask arguments
@click.option('--mask-path', help='Mask path (optional; None = auto-detect)', default=None, metavar='PATH')
@click.option('--mask-subdir', help='Mask sub-directory name', default='masks', metavar='NAME')
@click.option('--mask-suffix', help='Mask filename suffix (e.g. _mask)', default='', metavar='STR')
@click.option('--foreground-weight', help='Foreground weighting factor', default=5.0, type=float, metavar='FLOAT')

# Training arguments
@click.option('--kimg', help='Training budget in kilo-images', default=500, type=int, metavar='INT')
@click.option('--batch', help='Batch size', default=32, type=int, metavar='INT')
@click.option('--lr', help='Learning rate', default=0.0002, type=float, metavar='FLOAT')
@click.option('--gamma', help='R1 penalty coefficient', default=2.0, type=float, metavar='FLOAT')

# Device arguments
@click.option('--gpus', help='Number of GPUs', default=1, type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval (kimg)', default=50, type=int, metavar='INT')
@click.option('--tick', help='Progress log interval (kimg)', default=4, type=int, metavar='INT')
@click.option('--img-snap', help='Image snapshot interval (ticks)', default=10, type=int, metavar='INT')

def main(**kwargs):
    """Fine-tune R3GAN with foreground mask weighting."""
    opts = dnnlib.EasyDict(kwargs)

    # Set up output directory
    desc = f'finetune-{os.path.basename(opts.data).split(".")[0]}-fw{opts.foreground_weight}'
    opts.run_dir = os.path.join(opts.outdir, desc)
    os.makedirs(opts.run_dir, exist_ok=True)

    print('=' * 80)
    print(f'R3GAN Fine-tuning — Foreground Mask Weighting')
    print('=' * 80)
    print(f'Output directory:  {opts.run_dir}')
    print(f'Pre-trained model: {opts.resume}')
    print(f'Dataset:           {opts.data}')
    print(f'Foreground weight: {opts.foreground_weight}x')
    print(f'Training kimg:     {opts.kimg}')
    print('=' * 80)

    start_time = time.time()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Load pre-trained model
    print('\nLoading pre-trained model...')
    with open(opts.resume, 'rb') as f:
        resume_data = legacy.load_network_pkl(f)

    G = resume_data['G'].to(device).train().requires_grad_(True)
    D = resume_data['D'].to(device).train().requires_grad_(True)
    G_ema = resume_data['G_ema'].to(device).eval().requires_grad_(False)

    print(f'  Generator:     {sum(p.numel() for p in G.parameters())/1e6:.2f}M parameters')
    print(f'  Discriminator: {sum(p.numel() for p in D.parameters())/1e6:.2f}M parameters')

    # 2. Load dataset (with masks)
    print('\nLoading dataset...')
    training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset_with_mask.ImageFolderDatasetWithMask',
        path=opts.data,
        mask_path=opts.mask_path,
        mask_subdir=opts.mask_subdir,
        mask_suffix=opts.mask_suffix,
        use_labels=True,
        xflip=False,
        resolution=64
    )

    # Strip class_name before constructing dataset
    dataset_kwargs = {k: v for k, v in training_set_kwargs.items() if k != 'class_name'}
    training_set = ImageFolderDatasetWithMask(**dataset_kwargs)

    data_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=opts.batch,
        shuffle=True,
        num_workers=0,  # zip files are not safe with multiple workers
        pin_memory=True,
    )

    print(f'  Images:     {len(training_set)}')
    print(f'  Batch size: {opts.batch}')
    print(f'  Resolution: {training_set.resolution}x{training_set.resolution}')

    # 3. Initialise loss function (with mask support)
    print('\nInitialising loss function...')
    loss = R3GANLossWithMask(
        G=G,
        D=D,
        augment_pipe=None,  # No augmentation during fine-tuning
        foreground_weight=opts.foreground_weight
    )
    print(f'  Foreground weight: {opts.foreground_weight}x')

    # 4. Set up optimisers
    print('\nSetting up optimisers...')
    G_opt = torch.optim.Adam(G.parameters(), lr=opts.lr, betas=(0.0, 0.99))
    D_opt = torch.optim.Adam(D.parameters(), lr=opts.lr, betas=(0.0, 0.99))
    print(f'  Learning rate: {opts.lr}')
    print(f'  R1 penalty:    γ={opts.gamma}')

    # 5. Initialise logging
    print('\nInitialising logging...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_jsonl = open(os.path.join(opts.run_dir, 'stats.jsonl'), 'wt')
    stats_tfevents = None
    try:
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(opts.run_dir)
    except ImportError:
        print('  Skipping TensorBoard export')

    # 6. Prepare fixed noise for image snapshots
    print('\nPreparing image snapshot grid...')
    grid_size = (4, 4)
    grid_z = torch.randn([grid_size[0] * grid_size[1], G.z_dim], device=device)
    grid_c = torch.zeros([grid_size[0] * grid_size[1], training_set.label_dim], device=device)

    # Distribute labels evenly across the grid
    if training_set.label_dim > 0:
        labels_per_class = grid_size[0] * grid_size[1] // training_set.label_dim
        for i in range(training_set.label_dim):
            start_idx = i * labels_per_class
            end_idx = min(start_idx + labels_per_class, grid_size[0] * grid_size[1])
            grid_c[start_idx:end_idx, i] = 1

    # Generate initial image snapshot
    print('  Generating initial image snapshot...')
    try:
        with torch.no_grad():
            images = torch.cat([G_ema(grid_z[i:i+1], grid_c[i:i+1]).cpu() for i in range(len(grid_z))]).to(torch.float32).numpy()
        save_image_grid(images, os.path.join(opts.run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        print('  Initial snapshot saved.')
    except Exception as e:
        print(f'  Warning: Initial snapshot failed: {e}')
        print('  Training will continue; visualisation may be unavailable.')

    # 7. Training loop
    print(f'\nStarting fine-tuning for {opts.kimg} kimg...')
    print(f'Log interval: every {opts.tick} kimg (~{opts.tick * 1000 / opts.batch:.0f} batches)')
    print()

    total_kimg = opts.kimg
    tick_start_nimg = 0
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    cur_nimg = 0
    cur_tick = 0
    batch_idx = 0

    data_iter = iter(data_loader)

    try:
        while True:
            # Fetch next batch
            try:
                real_img, real_c, real_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                real_img, real_c, real_mask = next(data_iter)

            real_img = (real_img.to(device).to(torch.float32) / 127.5 - 1)
            real_c = real_c.to(device)
            real_mask = torch.nan_to_num(real_mask.to(device), nan=0.0, posinf=1.0, neginf=0.0)

            cur_batch = real_img.shape[0]
            gen_z = torch.randn([cur_batch, G.z_dim], device=device)

            # Discriminator step
            D_opt.zero_grad(set_to_none=True)
            loss.accumulate_gradients(
                phase='D',
                real_img=real_img,
                real_c=real_c,
                gen_z=gen_z,
                gamma=opts.gamma,
                gain=1.0,
                real_mask=real_mask
            )
            D_opt.step()

            # Generator step (mask passed so the masked adversarial branch is active)
            G_opt.zero_grad(set_to_none=True)
            loss.accumulate_gradients(
                phase='G',
                real_img=real_img,
                real_c=real_c,
                gen_z=gen_z,
                gamma=opts.gamma,
                gain=1.0,
                real_mask=real_mask,
            )
            G_opt.step()

            # Update EMA
            ema_nimg = 500
            ema_beta = 0.5 ** (cur_batch / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))

            cur_nimg += cur_batch
            batch_idx += 1

            # Inline progress (every 10 batches)
            if batch_idx % 10 == 0:
                print(f'\rProcessing: {cur_nimg/1000:.1f} / {total_kimg} kimg ({cur_nimg/1000/total_kimg*100:.1f}%)', end='', flush=True)

            # Log progress (every tick)
            done = (cur_nimg >= total_kimg * 1000)
            if (cur_nimg >= tick_start_nimg + opts.tick * 1000) or done:
                cur_tick += 1
                tick_end_time = time.time()

                # Clear inline progress line
                print('\r' + ' ' * 80 + '\r', end='')

                stats_collector.update()
                stats_dict = stats_collector.as_dict()

                # Build log fields (same format as original training)
                fields = []
                fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
                fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
                fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
                fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
                fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
                fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
                fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
                fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
                fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
                torch.cuda.reset_peak_memory_stats()

                training_stats.report0('Progress/lr', opts.lr)
                training_stats.report0('Progress/gamma', opts.gamma)
                training_stats.report0('Progress/foreground_weight', opts.foreground_weight)
                training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
                training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))

                print(' '.join(fields))

                # Update log files
                timestamp = time.time()
                if stats_jsonl is not None:
                    stats_jsonl.write(json.dumps(dict(stats_dict, timestamp=timestamp)) + '\n')
                    stats_jsonl.flush()
                if stats_tfevents is not None:
                    for name, value in stats_dict.items():
                        if isinstance(value, dict) and 'mean' in value:
                            stats_tfevents.add_scalar(name, value['mean'], int(cur_nimg))

                # Reset tick timer
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()
                maintenance_time = 0

            # Image snapshot
            if (cur_tick % opts.img_snap == 0 and cur_tick > 0) or done:
                try:
                    with torch.no_grad():
                        images = torch.cat([G_ema(grid_z[i:i+1], grid_c[i:i+1]).cpu() for i in range(len(grid_z))]).to(torch.float32).numpy()
                    save_image_grid(images, os.path.join(opts.run_dir, f'fakes{cur_nimg//1000:09d}.png'), drange=[-1,1], grid_size=grid_size)
                    print(f'Saved image snapshot: fakes{cur_nimg//1000:09d}.png')
                except Exception as e:
                    print(f'Warning: Image snapshot failed: {e}')

            # Model snapshot
            snapshot_ticks = opts.snap * 1000 / (opts.tick * 1000)
            if (cur_tick % max(1, round(snapshot_ticks)) == 0 and cur_tick > 0) or done:
                snapshot_pkl = os.path.join(opts.run_dir, f'network-snapshot-{cur_nimg//1000:09d}.pkl')

                snapshot_data = dict(
                    G=G,
                    D=D,
                    G_ema=G_ema,
                    training_set_kwargs=dict(training_set_kwargs),
                    cur_nimg=cur_nimg
                )

                def move_opt_state_to_cpu(state_dict):
                    """Recursively move optimiser state tensors to CPU."""
                    result = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.cpu()
                        elif isinstance(v, dict):
                            result[k] = move_opt_state_to_cpu(v)
                        else:
                            result[k] = v
                    return result

                snapshot_data['G_opt_state'] = move_opt_state_to_cpu(G_opt.state_dict())
                snapshot_data['D_opt_state'] = move_opt_state_to_cpu(D_opt.state_dict())

                # Move models to CPU before pickling
                for key, value in snapshot_data.items():
                    if isinstance(value, torch.nn.Module):
                        snapshot_data[key] = copy.deepcopy(value).eval().requires_grad_(False).cpu()

                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

                print(f'Saved snapshot: {snapshot_pkl}')

                del snapshot_data
                torch.cuda.empty_cache()

            if done:
                break

    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user.')
        print('Saving current state...')

        interrupt_pkl = os.path.join(opts.run_dir, f'network-snapshot-{cur_nimg//1000:09d}-interrupted.pkl')
        snapshot_data = dict(
            G=copy.deepcopy(G).eval().requires_grad_(False).cpu(),
            D=copy.deepcopy(D).eval().requires_grad_(False).cpu(),
            G_ema=copy.deepcopy(G_ema).eval().requires_grad_(False).cpu(),
            training_set_kwargs=dict(training_set_kwargs),
            cur_nimg=cur_nimg
        )
        with open(interrupt_pkl, 'wb') as f:
            pickle.dump(snapshot_data, f)
        print(f'Saved interrupt snapshot: {interrupt_pkl}')
        del snapshot_data

    except Exception as e:
        print(f'\n\nTraining error: {e}')
        import traceback
        traceback.print_exc()

    finally:
        if stats_jsonl is not None:
            stats_jsonl.close()
        if stats_tfevents is not None:
            stats_tfevents.close()

    print()
    print('Fine-tuning complete!')
    print(f'Results saved to: {opts.run_dir}')


if __name__ == "__main__":
    main()
