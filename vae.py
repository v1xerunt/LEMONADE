import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import utils
import vae_objectives
from normal_mixture import NormalMixture
from utils import get_mean_param
from utils import Constants

def extra_hidden_layer(hidden_dim):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))

class Enc(nn.Module):
    def __init__(self, data_dim, latent_dim, num_hidden_layers=1, hidden_dim=100):
        super(Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-1], -1))
        return self.fc21(e), F.softplus(self.fc22(e)) + Constants.eta

class Dec(nn.Module):
    def __init__(self, data_dim, latent_dim, num_hidden_layers, hidden_dim):
        super(Dec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.data_dim = data_dim
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)
        self.fc32 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.data_dim)  # reshape data
        return mu, F.softplus(self.fc32(d)).view_as(mu)

class VAE(nn.Module):
    def __init__(self, prior_dist=NormalMixture, posterior_dist=dist.Normal, likelihood_dist=dist.Normal, \
                 data_dim=512, latent_dim=256, num_hidden_layers=2, hidden_dim=256, disease_types=4):
        super(VAE, self).__init__()
        enc = Enc(data_dim, latent_dim, num_hidden_layers, hidden_dim)
        dec = Dec(data_dim, latent_dim, num_hidden_layers, hidden_dim)
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.disease_types = disease_types

        self._pz_mu, self._pz_logvar = self.init_pz(latent_dim, disease_types, False)
        self.prior_variance_scale = 1.
        print('p(z):')
        print(self.pz)
        print('q(z|x):')
        print(self.qz_x)

    @property
    def device(self):
        return self._pz_mu.device

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            pz = self.pz(*self.pz_params)
            if self.pz == torch.distributions.studentT.StudentT:
                pz._chi2 = torch.distributions.Chi2(pz.df)  # fix from rsample
            px_z_params = self.dec(pz.sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample())

        return get_mean_param(px_z_params)

    def forward(self, x, K=1, no_dec=False):
        qz_x = self.qz_x(*self.enc(x))
        # Number of samples to estimate ELBO
        zs = qz_x.rsample(torch.Size([K]))
        if no_dec:
            return qz_x, zs
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        mu = self._pz_mu.mul(1)
        scale = torch.sqrt(self.prior_variance_scale * self._pz_logvar.size(-1) * F.softmax(self._pz_logvar, dim=1))
        return mu, scale

    def init_pz(self, latent_dim, disease_types, learn_prior_variance):
        pz_mu = torch.randn((disease_types, latent_dim))
        pz_mu = nn.Parameter(pz_mu / torch.norm(pz_mu, 2, dim=1).unsqueeze(-1), requires_grad=False)
        pz_logvar = nn.Parameter(torch.zeros(disease_types, latent_dim), requires_grad=learn_prior_variance)
        return pz_mu, pz_logvar
