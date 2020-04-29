import torch
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn import metrics

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

import representation
import vae_objectives
from vae import VAE
import importlib
import classification
from classification import Classifications

test_ehr1 = pickle.load(open('./dataset/test_dataset1', 'rb'))
test_ehr2 = pickle.load(open('./dataset/test_dataset2', 'rb'))
test_ehr = test_ehr1+test_ehr2
test_type = pickle.load(open('./dataset/test_type', 'rb'))
test_label = pickle.load(open('./dataset/test_label', 'rb'))


def get_batch(idx, batch_size, x, y, types, device):
  if idx+batch_size <= len(x):
    end = idx+batch_size
  else:
    end = len(x)
  
  x_len = 0
  type_len = 0
  for i in range(idx,end):
    if len(x[i]) > x_len:
      x_len = len(x[i])
    if len(types[i]) > type_len:
      type_len = len(types[i])
  
  x_dim = len(x[0][0])
  batch_x = np.zeros((end-idx, x_len, x_dim))
  batch_type = np.zeros((end-idx, type_len)) - 1
  batch_len = []
  for i in range(idx, end):
    batch_x[i-idx, :len(x[i]), :] = x[i]
    batch_len.append(len(x[i]))
    batch_type[i-idx, :len(types[i])] = types[i]  
  batch_y = y[idx:end]
  
  batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
  batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
  batch_type = torch.tensor(batch_type, dtype=torch.long).to(device)
  batch_len = torch.tensor(batch_len, dtype=torch.long).to(device)
  
  return batch_x, batch_len, batch_type, batch_y

batch_size = 300
epochs = 50
lr=1e-3

data_dim = 1823
embedding_dim = 512
hidden_dim = 320
latent_dim = 360
vae_hidden_dim = 512
mlp_hidden_dim = 256
types = 10
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')

importlib.reload(classification)
importlib.reload(representation)
importlib.reload(vae_objectives)
lstm = representation.LSTM(data_dim=data_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
vae = VAE(data_dim=hidden_dim*2, latent_dim=latent_dim, hidden_dim=vae_hidden_dim, disease_types=types).to(device)
c_model = classification.Classifications(latent_dim, mlp_hidden_dim, types).to(device)
optimizer = torch.optim.Adam(list(c_model.parameters())+list(vae.parameters())+list(lstm.parameters()), lr=lr)

checkpoint = torch.load('./log/model')
lstm.load_state_dict(checkpoint['lstm'])
vae.load_state_dict(checkpoint['vae'])
c_model.load_state_dict(checkpoint['cmodel'])
optimizer.load_state_dict(checkpoint['optimizer'])
  
beta = []
nu = []
for i in range(types):
  beta.append(1/types)
  nu.append(1/types)
beta = torch.tensor(beta)
nu = torch.tensor(nu)

loss_his = []
sv_his = []
vae_his = []
pred_his = []
true_his = []

with torch.no_grad():
  lstm.eval()
  vae.eval()
  c_model.eval()

  for test_batch in range(0, len(test_ehr), batch_size):
    batch_x, batch_len, batch_type, batch_y = get_batch(test_batch, batch_size, test_ehr, test_label, test_type, device)

    ehr_representation = lstm(batch_x, batch_len, device)
    loss, sv_loss, lpx_z, kld, inc_kld, kld2, pred, true = vae_objectives.objective(vae, c_model, device, x=ehr_representation, mask=batch_type, types=types, label=batch_y, beta=beta, nu=nu, K=1, kappa=1.0, components=True)

    pred_his += list(pred)
    true_his += list(true)
    loss_his.append(loss.cpu().detach().numpy())
    sv_his.append(sv_loss.cpu().detach().numpy())
    vae_his.append((lpx_z.cpu().detach().numpy(), kld.cpu().detach().numpy(), inc_kld.cpu().detach().numpy(), kld2.cpu().detach().numpy()))

print('Classification loss: %.4f'%np.mean(sv_his))
vae_his = np.mean(np.array(vae_his), axis=0)
v_loss = -vae_his[0]+vae_his[1]+vae_his[2]+vae_his[3]
print('VAE Loss: %.4f'%v_loss)
print('--lpx_z: %.4f'%vae_his[0])
print('--kld: %.4f'%vae_his[1])
print('--inc_kld: %.4f'%vae_his[2])
print('--kld2: %.4f'%vae_his[3])


for i in range(types):
  cur_pred = np.array(pred_his)[:, i]
  cur_true = np.array(true_his)[:, i]
  auroc = roc_auc_score(cur_true, cur_pred)
  (precisions, recalls, thresholds) = metrics.precision_recall_curve(cur_true, cur_pred)
  auprc = metrics.auc(recalls, precisions)
  print('Disease %d, AUROC: %.4f, AUPRC: %.4f'%(i+1, auroc, auprc))
