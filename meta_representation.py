import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Surrogate(nn.Module):
    def __init__(self, data_dim, hidden_dim, types=10):
        super(Surrogate, self).__init__()
        self.alpha = torch.nn.Parameter(torch.randn(types))
        self.alpha.requires_grad = True

    def forward(self, l_ri, mask, device):
        p_r = []
        _max = torch.max(l_ri[mask != 0])
        alpha = torch.softmax(self.alpha)
        for i in range(len(l_ri)):
            if mask[i] == 0:
                continue
            else:
                ds = l_ri[i] - _max
                ds = torch.exp(ds)
                p_r.append((ds * alpha[i]))
        p_r = torch.stack(p_r).to(device) #k*n
        lpr = torch.sum(p_r, dim=0) #n
        lpr = _max + torch.log(lpr) - math.log(p_r.size(0)) #n
        lpr = torch.sum(lpr)
        return -lpr