from torch.optim import Adam

from beam import NeuralAlgorithm
from .model import FiLMUNet
from torch import nn
import torch
import math
import timm


class ExplicitGradientGenerativeAlgorithm(NeuralAlgorithm):


    def __init__(self, hparams, *args, **kwargs):

        g_net = FiLMUNet(hparams)
        f_net = timm.create_model('resnet18', pretrained=False, num_classes=1)
        super().__init__(hparams, *args, networks={'g': g_net, 'f': f_net}, **kwargs)
        self.loss = nn.MSELoss(reduction='none')

    def sample_eps(self, b, device=None):
        # Define the range in log-space
        log_min = math.log(self.hparams.eps_min)
        log_max = math.log(self.hparams.eps_max)

        # Sample uniformly in log-space
        log_sample = torch.rand(b, device=device) * (log_max - log_min) + log_min

        # Convert back to linear scale
        sample = torch.e ** log_sample
        return sample

    def generate_image(self, x, steps=1000, lr=1e-2):
        g_net = self.networks['g']
        f_net = self.networks['f']
        pred = f_net(x)
        eps = pred[:, 0].expand(1, 10)
        best = x.detach().clone()
        b = len(x)
        best_eps = float('inf') * torch.ones(b, device=x.device)
        x = nn.Parameter(data=x)
        opt = Adam([x], lr=.01)
        with torch.no_grad():
            for i in range(steps):
                g = g_net(x, eps)
                x.grad = -g
                opt.step()
                pred = f_net(x)
                eps = pred[:, 0].expand(1, 10)
                for i, e in enumerate(pred[:, 0]):
                    if e < best_eps[i]:
                        best[i] = x[i]
                        best_eps[i] = e
        return best

    def train_iteration(self, sample=None, label=None, index=None, counter=None, subset=None, training=True, **kwargs):

        g_net = self.networks['g']
        f_net = self.networks['f']
        x = sample['x']
        # y = sample['y']
        b = len(x)

        # eps = self.sample_eps(b, device=x.device)
        eps = torch.rand(b, device=x.device)
        noise = torch.randn(*x.shape, device=x.device)

        eps_context = eps.unsqueeze(1).expand(-1, self.hparams.context_dim)
        eps_image = eps.view(-1, 1, 1, 1)
        x_perturbed = x * (1 - eps_image) + eps_image * noise
        g_x_perturbed = g_net(x_perturbed, eps_context)
        f_pred = f_net(x_perturbed)

        y = torch.ones(b, device=x.device)
        y_perturbed = 1 - eps

        eps_pred = f_pred[:, 0]

        diff = x - x_perturbed

        # y = y_perturbed + g_x_perturbed * (x - x_perturbed) + residual
        grad_prod = (g_x_perturbed * diff).sum(dim=(1, 2, 3))
        loss_g = self.loss(grad_prod, y - y_perturbed)
        loss_eps = self.loss(eps_pred, eps)

        self.report_scalar('objective', (torch.abs(grad_prod) / eps).mean())
        self.apply({'g_loss': loss_g, 'eps_loss': loss_eps})


