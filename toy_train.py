import torch
import representation
import vae_objectives
from vae import VAE
from classification import MLP

ehr = torch.randn((10, 15, 23))  #Batch_size * time_step * features
label = torch.tensor([0,0,1,0,1,0,0,1,0,1], dtype=torch.float32)
mask = torch.tensor([0,1,2,0,3,4,4,0,3,2])
device = 'cpu'

lstm = representation.LSTM(data_dim=23, hidden_dim=64)
ehr_representation = lstm(ehr, device)

vae = VAE(data_dim=128, latent_dim=64, hidden_dim=128, disease_types=4)
mlp = MLP(data_dim=64, hidden_dim=64)
beta = torch.tensor([0.2,0.2,0.2,0.2,0.2])
nu = torch.tensor([0.2,0.2,0.2,0.2,0.2])
loss, sv_loss, lpx_z, kld, inc_kld, kld2 = vae_objectives.objective(vae, mlp, 'cpu', x=ehr_representation, mask=mask, label=label, beta=beta, nu=nu, K=1, kappa=1.0, components=True)
print('Loss: %.4f' % loss)
print('Supervised loss: %.4f' % sv_loss)
print('log(p(x|z)): %.4f' % lpx_z)
print('KL(q(z|x), p(z)): %.4f' % kld)
print('KL(p(z), q(z)): %.4f' % inc_kld)
print('KL(q(z), N(z)): %.4f'%kld2)