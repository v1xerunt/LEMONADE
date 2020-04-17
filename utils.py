import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
import numpy as np

# Classes
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)

def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    torch.save(model.state_dict(), filepath)


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    return params[1] if params[0].dim() == 0 else params[0]


def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()

def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY

def kl_divergence(p, q, samples=None):
    if has_analytic_kl(type(p), type(q)):
        return dist.kl_divergence(p, q)
    else:
        ent = -p.log_prob(samples)
        return (-ent - q.log_prob(samples)).mean(0)

def kld_inc(pz, qz_x):
    B, D = qz_x.loc.shape
    _zs = pz.rsample(torch.Size([B]))
    lpz = pz.log_prob(_zs).sum(-1).squeeze(-1)
    _zs = _zs.expand(B, B, D)
    lqz = log_mean_exp(qz_x.log_prob(_zs).sum(-1), dim=1)
    inc_kld = lpz - lqz
    inc_kld = inc_kld.mean(0, keepdim=True).expand(1, B)
    return inc_kld.mean(0).sum() / B