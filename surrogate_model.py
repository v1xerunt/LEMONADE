import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from utils import Constants

class Surrogate(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_hidden_layers=1, types=10):
        super(Surrogate, self).__init__()
        self.types = types

        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        self.dist_params = nn.Sequential(*modules)
        self.nn_params = nn.ModuleList([dist_params for i in range(types)])
        self.mus = nn.ModuleList([nn.Linear(hidden_dim, data_dim) for i in range(types)])
        self.sigmas = nn.Sequential()
        self.sigmas = nn.ModuleList([nn.Linear(hidden_dim, data_dim) for i in range(types)])

    def forward(self, z, t, device):
        batch_size, data_dim = z.size()

        mu = []
        sigma = []
        for i in range(self.types):
            tmp_t = torch.sum(t == i,dim=-1).bool()
            if torch.sum(tmp_t) == 0:
                mu.append(-1)
                sigma.append(-1)
                continue
            tmp_z = z[:, tmp_t]
            e = self.nn_params[i](tmp_z)
            mu.append(self.mus(e))
            sigma.append(F.softplus(self.sigmas(e)) + Constants.eta)

        return mu, sigma