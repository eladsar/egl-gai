from beam import NeuralAlgorithm
from .model import FiLMUNet
from torch import nn
import torch
import math


class ExplicitGradientGenerativeAlgorithm(NeuralAlgorithm):


    def __init__(self, hparams, *args, **kwargs):

        net = FiLMUNet(hparams)
        super().__init__(hparams, *args, networks=net, **kwargs)
        self.loss = nn.MSELoss()

    def sample_eps(self, b, device=None):
        # Define the range in log-space
        log_min = math.log(self.hparams.eps_min)
        log_max = math.log(self.hparams.eps_max)

        # Sample uniformly in log-space
        log_sample = torch.rand(b, device=device) * (log_max - log_min) + log_min

        # Convert back to linear scale
        sample = torch.e ** log_sample
        return sample


    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):

        net = self.networks['net']
        x = sample['x']
        y = sample['y']
        b = len(x)

        eps = self.sample_eps(b, device=x.device)
        noise = torch.randn(*x.shape, device=x.device)

        eps_context = eps.unsqueeze(1).expand(-1, self.hparams.context_dim)
        eps_image = eps.view(-1, 1, 1, 1)
        x_perturbed = x * (1 - eps_image) + eps_image * noise
        g_x_perturbed = net(x_perturbed, eps_context)

        y = torch.ones(b, device=x.device)
        y_perturbed = 1 - eps

        diff = x - x_perturbed

        # y_perturbed = y + g_x_perturbed * diff + residual
        loss = self.loss((g_x_perturbed * diff).sum(dim=(1, 2, 3)), y - y_perturbed)
        self.apply(loss)


