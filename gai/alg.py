from beam import NeuralAlgorithm
from .model import FiLMUNet
from torch import nn
import torch


class ExplicitGradientGenerativeAlgorithm(NeuralAlgorithm):


    def __init__(self, *args, **kwargs):

        net = FiLMUNet()
        super().__init__(*args, networks=net, **kwargs)
        self.loss = nn.MSELoss()

    def sample_eps(self, b):
        # Define the range in log-space
        log_min = torch.log(torch.tensor(self.hparams.eps_min))
        log_max = torch.log(torch.tensor(self.hparams.eps_max))

        # Sample uniformly in log-space
        log_sample = torch.rand(b) * (log_max - log_min) + log_min

        # Convert back to linear scale
        sample = torch.e ** log_sample
        return sample


    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):

        net = self.networks['net']
        x = sample['x']
        y = sample['y']
        b = len(x)

        eps_vals = self.sample_eps(b).unsqueeze(1).view(-1, self.hparams.context_dim)
        eps = torch.randn(*x.shape) * eps_vals.view(-1, 1, 1, 1)

        x_perturbed = x - eps
        g_x_perturbed = net(x_perturbed, eps)

        y = torch.ones(b, device=x.device)
        y_perturbed = 1 - torch.clamp_max(sample / torch.abs(x).mean(dim=(1, 2, 3)), 1)

        # residual = y - y_perturbed - (g_x_perturbed * eps).sum(dim=1)
        loss = self.loss((g_x_perturbed * eps).sum(dim=1), y - y_perturbed)
        self.apply(loss)


