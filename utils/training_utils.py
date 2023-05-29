""" 
History
    - 230419 : MINSU , init
        - Copy <- ZSSGAN/utils/training_utils.py
    - 230423 : MINSU , add
        - add def save_image_grid 
""" 
import torch
import math
import random
from torchvision import utils
import os

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths



# from NADA ZSSGAN/utils/file_utils.py 
def save_images(images: torch.Tensor, output_dir: str, filename, nrows: int, ) -> None:
    utils.save_image(
        images,
        os.path.join(output_dir, f"{filename}.png"),
        nrow=nrows,
        normalize=True,
        range=(-1, 1),
    )
