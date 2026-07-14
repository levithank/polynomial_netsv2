import torch
import torch.nn as nn
import os
import json
from torch import optim
from torch.autograd import grad as torch_grad

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe (works on Colab / servers)
import matplotlib.pyplot as plt

from model import Discriminator
from model import Generator
from logger import Logger

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_score(base_path, entry):
    save_file = os.path.join(base_path, "scores.json")
    with open(save_file, 'a+') as fp:
        json.dump(entry, fp, indent=4)
        fp.write("\n")


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_numpy(x):
    if x.is_cuda:
        x = x.data.cpu()
    return x.detach().numpy()


class Solver(object):
    def __init__(self, config, data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None

        self.pc_name = config.pc_name
        self.base_path = config.base_path
        self.data_loader = data_loader
        self.num_epochs = config.num_epochs
        self.sample_size = config.sample_size
        self.logs_path = config.logs_path
        self.save_every = config.save_every
        self.activation_fn = config.activation_fn
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

        # ------------------------------------------------------------------
        # Map the old image-GAN config onto the 2D point model.
        #
        # The point Generator expects:
        #   z_dim, hidden_dim, num_orders, activation_fn, bound_output
        #
        # We reuse g_layers[0] as z_dim (it was 100 = noise size in the
        # DCGAN config). hidden_dim / num_orders / bound_output fall back
        # to sensible defaults if your argparse doesn't define them yet.
        # ------------------------------------------------------------------
        self.z_dim = config.g_layers[0]
        self.hidden_dim = getattr(config, 'hidden_dim', 128)
        self.num_orders = getattr(config, 'num_orders', 3)
        self.bound_output = getattr(config, 'bound_output', False)
        self.spectral_norm = config.spectral_norm

        self.build_model()

    def build_model(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.generator = Generator(
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            num_orders=self.num_orders,
            activation_fn=self.activation_fn,
            bound_output=self.bound_output
        )
        self.discriminator = Discriminator(
            input_dim=2,
            hidden_dim=self.hidden_dim,
            use_spectral_norm=self.spectral_norm
        )

        self.g_optimizer = optim.Adam(
            self.generator.parameters(), self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), self.lr,
            betas=(self.beta1, self.beta2)
        )
        self.logger = Logger(self.logs_path)

        self.gen_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.disc_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print("Generator params: {}".format(self.gen_params))
        print("Discriminator params: {}".format(self.disc_params))
        print("Total params: {}".format(self.gen_params + self.disc_params))

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def reset_grad(self):
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    def gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        alpha = to_cuda(alpha)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = to_cuda(interpolated)
        interpolated.requires_grad_(True)

        prob_interpolated = self.discriminator(interpolated)

        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=to_cuda(torch.ones(prob_interpolated.size())),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
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

                # Some datasets yield (points, labels); keep only points.
                if isinstance(data, (list, tuple)):
                    data = data[0]

                data = data.type(torch.FloatTensor)
                data = to_cuda(data)
                batch_size = data.size(0)

                real_labels = to_cuda(torch.ones(batch_size, 1))
                fake_labels = to_cuda(torch.zeros(batch_size, 1))

                # ---------------- train Discriminator ----------------
                outputs_real = self.discriminator(data)

                # NOTE: the point Generator expects z of shape
                # [batch, z_dim] -- NOT [batch, z_dim, 1, 1].
                z = to_cuda(torch.randn(batch_size, self.z_dim))
                fake_data = self.generator(z)
                outputs_fake = self.discriminator(fake_data.detach())

                if self.loss == 'original':
                    d_loss_real = self.criterion(outputs_real, real_labels)
                    d_loss_fake = self.criterion(outputs_fake, fake_labels)
                    d_loss = d_loss_real + d_loss_fake
                elif self.loss == 'wgan-gp':
                    gp = self.gradient_penalty(data, fake_data)
                    d_loss = -outputs_real.mean() + outputs_fake.mean() + gp
                else:
                    raise ValueError("Unknown loss: {}".format(self.loss))

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # ------------------ train Generator -------------------
                z = to_cuda(torch.randn(batch_size, self.z_dim))
                fake_data = self.generator(z)
                outputs_fake = self.discriminator(fake_data)

                if self.loss == 'original':
                    g_loss = self.criterion(outputs_fake, real_labels)
                elif self.loss == 'wgan-gp':
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
                        z = to_cuda(torch.randn(self.sample_size, self.z_dim))
                        samples = to_numpy(self.generator(z))
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
                        z = to_cuda(torch.randn(2048, self.z_dim))
                        val_points = to_numpy(self.generator(z))
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
        self.generator = Generator(
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            num_orders=self.num_orders,
            activation_fn=self.activation_fn,
            bound_output=self.bound_output
        )
        self.generator.load_state_dict(
            torch.load(self.ckpt_gen_path, map_location='cpu'))
        if torch.cuda.is_available():
            self.generator.cuda()
        self.generator.eval()

        with torch.no_grad():
            z_samples = to_cuda(torch.randn(n_samples, self.z_dim))
            generated = to_numpy(self.generator(z_samples))

        os.makedirs('./saved', exist_ok=True)
        np.save('./saved/generated_samples.npy', generated)
        np.save('./saved/z_samples.npy', to_numpy(z_samples))