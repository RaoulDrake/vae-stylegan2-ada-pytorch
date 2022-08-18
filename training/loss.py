# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import dnnlib

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------


class StyleGAN2VAELoss(StyleGAN2Loss):
    def __init__(
        self, device, G_mapping_implicit, G_mapping_explicit, G_synthesis, D, E, vgg16=None,
        augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
        pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
        pixel_loss_weight: float = 0.0, perceptual_loss_weight: float = 1.0,
        kld_loss_weight: float = 0.0001
    ):
        super().__init__(
            device=device, G_mapping=G_mapping_implicit, G_synthesis=G_synthesis, D=D,
            augment_pipe=augment_pipe, style_mixing_prob=style_mixing_prob, r1_gamma=r1_gamma,
            pl_batch_shrink=pl_batch_shrink, pl_decay=pl_decay, pl_weight=pl_weight
        )
        self.G_mapping_explicit = G_mapping_explicit
        self.E = E
        self.vgg16 = vgg16
        self.pixel_loss_weight = pixel_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.kld_loss_weight = kld_loss_weight

    def run_G(self, z, c, sync, implicit=True):
        mapping = self.G_mapping
        if not implicit:
            mapping = self.G_mapping_explicit
        with misc.ddp_sync(mapping, sync):
            ws = mapping(z, c)
            if self.style_mixing_prob > 0 and implicit:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_E(self, img, c, sync):
        with misc.ddp_sync(self.E, sync):
            z, mean, log_var = self.E(img, c)
        return z, mean, log_var

    def run_vgg16(self, img):
        # Scale dynamic range from [-1,1] to [0,255].
        img = (img + 1) * (255 / 2)
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if img.shape[2] > 256:
            img = torch.nn.functional.interpolate(img, size=(256, 256), mode='area')
        features = self.vgg16(img, resize_images=False, return_lpips=True)
        return features

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        do_Emain = (phase in ['Emain', 'Eboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)

        if not do_Emain:
            super().accumulate_gradients(
                phase=phase, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain
            )
        else:
            # VAE
            with torch.autograd.profiler.record_function('Emain_forward'):
                # Reconstruct
                real_img_tmp = real_img.detach().requires_grad_(False)
                z, mean, log_var = self.run_E(real_img_tmp, real_c, sync=sync)
                recon_img, _recon_ws = self.run_G(z, real_c, implicit=False, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                # VAE Loss
                perceptual_loss = 0.0
                if self.vgg16 is not None:
                    # Get features
                    real_features = self.run_vgg16(real_img_tmp)
                    recon_features = self.run_vgg16(recon_img)
                    # Perceptual loss
                    perceptual_loss = (real_features - recon_features).square().sum(1).mean()
                # Pixel loss
                pixel_loss = torch.mean((real_img_tmp - recon_img).square())  # , dim=[1, 2, 3])
                # KLD loss
                kld_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + torch.square(mean) - log_var - 1, dim=1))
                training_stats.report('Loss/E/pixel_loss', pixel_loss)
                training_stats.report('Loss/E/pixel_loss_weighted', self.pixel_loss_weight * pixel_loss)
                training_stats.report('Loss/E/perceptual_loss', perceptual_loss)
                training_stats.report('Loss/E/perceptual_loss_weighted', self.perceptual_loss_weight * perceptual_loss)
                training_stats.report('Loss/E/kld_loss', kld_loss)
                training_stats.report('Loss/E/kld_loss_weighted', self.kld_loss_weight * kld_loss)
                training_stats.report('Loss/E/kld_loss_weight', self.kld_loss_weight)
                training_stats.report('Loss/E/loss', (
                    self.pixel_loss_weight * pixel_loss +
                    self.perceptual_loss_weight * perceptual_loss +
                    self.kld_loss_weight * kld_loss
                ))
            with torch.autograd.profiler.record_function('Emain_backward'):
                (
                    self.pixel_loss_weight * pixel_loss +
                    self.perceptual_loss_weight * perceptual_loss +
                    self.kld_loss_weight * kld_loss
                ).mul(gain).backward()

#----------------------------------------------------------------------------


class VAELoss(StyleGAN2VAELoss):
    def __init__(
        self, device, G_mapping, G_synthesis, E, vgg16=None,
        pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
        pixel_loss_weight: float = 0.0, perceptual_loss_weight: float = 1.0,
        kld_loss_weight: float = 0.0001
    ):
        super().__init__(
            device=device,
            G_mapping_implicit=G_mapping, G_mapping_explicit=G_mapping, G_synthesis=G_synthesis,
            D=None, E=E, vgg16=vgg16,
            pl_batch_shrink=pl_batch_shrink, pl_decay=pl_decay, pl_weight=pl_weight,
            pixel_loss_weight=pixel_loss_weight, perceptual_loss_weight=perceptual_loss_weight,
            kld_loss_weight=kld_loss_weight
        )

#----------------------------------------------------------------------------
