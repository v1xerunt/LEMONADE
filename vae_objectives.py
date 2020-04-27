import torch
import torch.nn.functional as F
import torch.distributions as dist
import math
from numpy import prod
from utils import kl_divergence, log_mean_exp
import utils

def compute_microbatch_split(x, K):
    B = x.size(0)
    S = int(1e8 / (K * prod(x.size()[1:])))  # 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)

def objective(vae_model, c_model, device, x, mask, types, label, beta, nu, K=10, kappa=1.0, components=False):
    """Computes E_{p(x)}[ELBO_{\alpha,\beta}] """
    types = vae_model.disease_types
    qz_x, px_z, zs = vae_model(x, K)
    
    # compute supervised loss
    pred = c_model(zs.squeeze(0), device)
    bce = torch.nn.BCELoss(reduction='none')
    onehot_label = torch.zeros((x.size(0), types), dtype=torch.float32)
    for i in range(len(x)):
      for j in range(len(mask[i])):
        onehot_label[i][mask[i][j]] = 1
    supervised_loss = bce(pred, onehot_label)

    # compute vae loss
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1).sum(-1)
    pz = vae_model.pz(*vae_model.pz_params)
    kld = kl_divergence(qz_x, pz, samples=zs).sum(-1)

    # compute kl(p(z), q(z))
    B, D = qz_x.loc.shape
    _zs = pz.rsample(torch.Size([B]))
    lpz = pz.log_prob(_zs).sum(-1).squeeze(-1)

    _zs_expand = _zs.expand(B, B, D)
    lqz = qz_x.log_prob(_zs_expand).sum(-1)  #B*B
    
#    qz = []
#    _max = torch.max(lqz)
#    for i in range(types):
#        tmp_mask = torch.sum(mask == i,dim=-1).bool()
#        ds = lqz[:, tmp_mask] - _max
#        qz_j = torch.exp(ds)  #B*k
#        qz.append((qz_j * beta[i]))
#    qz = torch.cat(qz, dim=1).to(device)
#    lqz = torch.sum(qz, dim=1)
#    lqz = _max + torch.log(lqz) - math.log(qz.size(1))
    lqz = log_mean_exp(lqz, dim=1)  
  
    inc_kld = lpz - lqz
    inc_kld = inc_kld.mean(0, keepdim=True).expand(1, B)
    inc_kld = inc_kld.mean(0).sum() / B
    
    # compute kl(q(z), N(z))
    _zs = qz_x.rsample(torch.Size([B]))  #B*B*D
    lqz = qz_x.log_prob(_zs)  #B*B*D
    kld2 = []
    for i in range(types):
        tmp_mask = torch.sum(mask == i,dim=-1).bool()
        if torch.sum(tmp_mask) == 0:
          continue
        tmp_zs = _zs[:, tmp_mask, :]  #B*k*D
        lnj = dist.Normal(vae_model.pz_params[0][i], vae_model.pz_params[1][i]).log_prob(tmp_zs)
        _kl = (tmp_zs - lnj).mean(0) #k*D
        kld2.append((_kl * nu[i]).mean(0))
    kld2 = torch.stack(kld2).to(device)
    kld2 = torch.sum(kld2)
    
    obj = supervised_loss.sum(dim=-1) - lpx_z + kld.mean() + kappa * inc_kld + kld2
    return obj.mean() if not components else (obj.mean(), supervised_loss.mean(dim=0), lpx_z.mean(), kld.mean(), inc_kld.mean(), kld2.mean(), pred.cpu().detach().numpy(), onehot_label.cpu().detach().numpy())