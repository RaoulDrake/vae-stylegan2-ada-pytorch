# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

from typing import Optional, List

import copy
import json
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from training.dataset import ImageFolderDataset
from generate import num_range


def project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    target_label: torch.Tensor = None,  # one-hot label
    num_steps: int = 1000,
    w_avg_samples: int = 10000,
    initial_learning_rate: float = 0.1,
    initial_noise_factor: float = 0.05,
    lr_rampdown_length: float = 0.25,
    lr_rampup_length: float = 0.05,
    noise_ramp_length: float = 0.75,
    regularize_noise_weight: float = 1e5,
    verbose: bool = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    target_labels = target_label.unsqueeze(0).to(device).to(torch.float32)
    target_labels = target_labels.repeat(w_avg_samples, 1)
    w_samples = G.mapping(
        torch.from_numpy(z_samples).to(device), target_labels
    )  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Init distance and loss
    dist = 0
    loss = 0

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0:
            logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    result_dict = {'num_steps': num_steps, 'dist': float(dist), 'loss': float(loss)}

    return w_out.repeat([1, G.mapping.num_ws, 1]), result_dict

# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--classes', type=num_range, help='Restrict output to these classes', default='0-129', show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress',
              type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--start-img-per-class', 'start_img_per_class', help='Starts with the i-th img per class',
              type=int, default=0, show_default=True)
@click.option('--stop-img-per-class', 'stop_img_per_class', help='Stops with the i-th img per class', type=int,
              default=10, show_default=True)
def run_projection(
    network_pkl: str,
    outdir: str,
    data,  # <path>
    save_video: bool,
    seed: int,
    classes: Optional[List[int]],
    num_steps: int,
    start_img_per_class: int,
    stop_img_per_class: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore

    # Load Tsinghua dataset
    # data_path = "/mnt/e/Data/CVDL_Datasets/tsinghua_dogs_bnd_box_crop__style_gan_2_ada.zip"
    tsinghua_dataset = ImageFolderDataset(path=data, use_labels=True, max_size=None, xflip=False)
    imgs_and_one_hot_labels_per_label = {}
    for img, label_one_hot in tsinghua_dataset:
        label_int = int(np.argmax(label_one_hot))
        if label_int not in imgs_and_one_hot_labels_per_label:
            imgs_and_one_hot_labels_per_label[label_int] = []
        imgs_and_one_hot_labels_per_label[label_int].append((img, label_one_hot))

    for label_int, imgs_and_one_hot_labels in imgs_and_one_hot_labels_per_label.items():
        if label_int in classes:
            for i, (img, label_one_hot) in enumerate(imgs_and_one_hot_labels[start_img_per_class:stop_img_per_class]):
                print(f'Projecting image {start_img_per_class + i} of class {label_int}...')
                # Optimize projection.
                start_time = perf_counter()
                projected_w_steps, result_dict = project(
                    G,
                    target=torch.tensor(img, device=device),  # pylint: disable=not-callable
                    target_label=torch.tensor(label_one_hot, device=device),  # pylint: disable=not-callable
                    num_steps=num_steps,
                    device=device,
                    verbose=True
                )
                print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

                # Render debug output: optional video and projected image and W vector.
                img_outdir = os.path.join(outdir, str(label_int), str(start_img_per_class + i))
                os.makedirs(img_outdir, exist_ok=True)
                if save_video:
                    video = imageio.get_writer(f'{img_outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
                    print(f'Saving optimization progress video "{img_outdir}/proj.mp4"')
                    for projected_w in projected_w_steps:
                        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                        synth_image = (synth_image + 1) * (255/2)
                        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                        video.append_data(np.concatenate([img, synth_image], axis=1))
                    video.close()

                # Save final projected frame and W vector and result_dict.
                PIL.Image.fromarray(img.transpose([1, 2, 0])).save(f'{img_outdir}/target.png')
                projected_w = projected_w_steps[-1]
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(synth_image, 'RGB').save(f'{img_outdir}/proj.png')
                np.savez(f'{img_outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
                with open(f'{img_outdir}/result_dict.json', 'w', encoding='utf8') as f:
                    f.write(json.dumps(result_dict))

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
