import torch
import torch.distributions as dist
from utils import log_mean_exp
import numpy as np

def get_surrogate_objective(s_model, qz_x, z, mask, types, n, rho, device):
    mu, sigma = s_model(z, mask, device)
    kld = []
    l_r = []
    for i in range(types):
        if mu[i] == -1:
            continue
        tmp_mask = torch.sum(mask == i, dim=-1).bool()
        tmp_mu = mu[i] #k * D
        tmp_sigma = sigma[i]

        ri = dist.Normal(tmp_mu, tmp_sigma)
        z_ri = ri.rsample(torch.size([n])) #n * K * D
        l_ri = ri.log_prob(z_ri).sum(-1) #n*K
        l_qi = qz_x.log_prob(z_ri).sum(-1) #n*K
        l_qi = torch.logsumexp(l_qi).expand_as(l_ri) #n*K
        kld.append(torch.mean(l_ri - l_qi, dim=1))
        l_r.append(torch.mean(l_ri, dim=1))
    
    l_r = torch.logsumexp(torch.stack(l_r), dim=0) #n
    kld = torch.sum(torch.mean(torch.stack(kld))) #1
    obj = kld * rho - torch.sum(l_r)
    return obj, kld, l_r

tmp = torch.tensor([3, 4, 7, 5])
print(torch.max(tmp[tmp!=7]))