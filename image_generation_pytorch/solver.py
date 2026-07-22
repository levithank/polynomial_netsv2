import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import grad as torch_grad

import matplotlib
matplotlib.use('Agg')  # headless-safe (works on Colab / servers)
import matplotlib.pyplot as plt

from model import Discriminator, Generator
from logger import Logger


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_numpy(x):
    return x.detach().cpu().numpy()


class Solver(object):
    def __init__(self, config, data_loader):
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.sample_size = config.sample_size
        self.logs_path = config.logs_path
        self.save_every = config.save_every
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.validation_step = config.validation_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.ckpt_gen_path = config.ckpt_gen_path
        self.gp_weight = config.gp_weight
        self.loss = config.loss
        self.seed = config.seed
        self.criterion = nn.BCEWithLogitsLoss()

        self.build_model()

    def build_model(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Architecture is defined entirely by the defaults in model.py.
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.z_dim = self.generator.z_dim

        print("\nGenerator configuration:")
        print("  z_dim:", self.generator.z_dim)
        print("  hidden_dim:", self.generator.hidden_dim)
        print("  num_orders:", self.generator.num_orders)
        print("  activation:", self.generator.activation)

        print("\nDiscriminator configuration:")
        print("  input_dim:", self.discriminator.input_dim)
        print("  hidden_dim:", self.discriminator.hidden_dim)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(), self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.logger = Logger(self.logs_path)

        gen_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        disc_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print("Generator params: {}".format(gen_params))
        print("Discriminator params: {}".format(disc_params))
        print("Total params: {}".format(gen_params + disc_params))

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def sample_z(self, n):
        """Cauchy prior -> theta = 2*arctan(z) is uniform on the circle."""
        u = torch.rand(n, self.z_dim)
        return to_cuda(torch.tan(math.pi * (u - 0.5)).clamp(-1e4, 1e4))

    def reset_grad(self):
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    def gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        alpha = to_cuda(torch.rand(batch_size, 1))
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = interpolated.detach().requires_grad_(True)

        prob_interpolated = self.discriminator(interpolated)

        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True
        )[0]

        gradients_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def save_scatter(self, points, path, real_points=None):
        """Save a scatter plot of generated (and optionally real) points."""
        plt.figure(figsize=(5, 5))
        if real_points is not None:
            plt.scatter(real_points[:, 0], real_points[:, 1],
                        s=8, alpha=0.4, label='real')
        plt.scatter(points[:, 0], points[:, 1],
                    s=8, alpha=0.6, label='generated')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def train(self):
        total_step = len(self.data_loader)
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.data_loader):
                data = to_cuda(data)
                batch_size = data.size(0)

                # ---------------- train Discriminator ----------------
                outputs_real = self.discriminator(data)

                z = self.sample_z(batch_size)
                fake_data = self.generator(z)
                outputs_fake = self.discriminator(fake_data.detach())

                if self.loss == 'original':
                    real_labels = to_cuda(torch.ones(batch_size, 1))
                    fake_labels = to_cuda(torch.zeros(batch_size, 1))
                    d_loss = (self.criterion(outputs_real, real_labels)
                              + self.criterion(outputs_fake, fake_labels))
                elif self.loss == 'wgan-gp':
                    gp = self.gradient_penalty(data, fake_data)
                    d_loss = -outputs_real.mean() + outputs_fake.mean() + gp
                else:
                    raise ValueError("Unknown loss: {}".format(self.loss))

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # ------------------ train Generator -------------------
                z = self.sample_z(batch_size)
                fake_data = self.generator(z)
                outputs_fake = self.discriminator(fake_data)

                if self.loss == 'original':
                    real_labels = to_cuda(torch.ones(batch_size, 1))
                    g_loss = self.criterion(outputs_fake, real_labels)
                else:
                    g_loss = -outputs_fake.mean()

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # ---------------------- logging -----------------------
                if (i + 1) % self.log_step == 0:
                    print('Epoch [{0:d}/{1:d}], Step [{2:d}/{3:d}], '
                          'd_loss: {4:.4f}, g_loss: {5:.4f}'.format(
                              epoch + 1, self.num_epochs, i + 1,
                              total_step, d_loss.item(), g_loss.item()))

                    info = {
                        'd_loss': d_loss.item(),
                        'g_loss': g_loss.item(),
                    }
                    for tag, value in info.items():
                        self.logger.scalar_summary(
                            tag, value, epoch * total_step + i + 1)

                # -------------------- sampling ------------------------
                if (i + 1) % self.sample_step == 0:
                    with torch.no_grad():
                        samples = to_numpy(self.generator(self.sample_z(self.sample_size)))
                    fig_path = os.path.join(
                        self.sample_path,
                        "epoch_{}_{}.png".format(epoch + 1, i + 1))
                    self.save_scatter(samples, fig_path,
                                      real_points=to_numpy(data))

                # ------------------- validation -----------------------
                # IS / FID are image metrics and don't apply to 2D points,
                # so we save the raw generated points instead.
                if (i + 1) % self.validation_step == 0:
                    with torch.no_grad():
                        val_points = to_numpy(self.generator(self.sample_z(2048)))
                    npy_path = os.path.join(
                        self.model_path,
                        '{}_{}_val_points.npy'.format(epoch + 1, i + 1))
                    np.save(npy_path, val_points)

            if (epoch + 1) % self.save_every == 0:
                g_path = os.path.join(self.model_path, 'generator-{}.pkl'.format(epoch + 1))
                d_path = os.path.join(self.model_path, 'discriminator-{}.pkl'.format(epoch + 1))
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)

    def sample(self, n_samples):
        self.generator.load_state_dict(
            torch.load(self.ckpt_gen_path, map_location='cpu'))
        self.generator.eval()

        with torch.no_grad():
            z_samples = self.sample_z(n_samples)
            generated = to_numpy(self.generator(z_samples))

        os.makedirs('./saved', exist_ok=True)
        np.save('./saved/generated_samples.npy', generated)
        np.save('./saved/z_samples.npy', to_numpy(z_samples))