import torch
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

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

train_ehr1 = pickle.load(open('./dataset/train_dataset1', 'rb'))
train_ehr2 = pickle.load(open('./dataset/train_dataset2', 'rb'))
train_ehr3 = pickle.load(open('./dataset/train_dataset3', 'rb'))
train_ehr = train_ehr1+train_ehr2+train_ehr3
train_type = pickle.load(open('./dataset/train_type', 'rb'))
train_label = pickle.load(open('./dataset/train_scoring', 'rb'))

valid_ehr = pickle.load(open('./dataset/valid_dataset', 'rb'))
valid_type = pickle.load(open('./dataset/valid_type', 'rb'))
valid_label = pickle.load(open('./dataset/valid_label', 'rb'))


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
  batch_x = np.zeros((batch_size, x_len, x_dim))
  batch_type = np.zeros((batch_size, type_len)) - 1
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

batch_size = 128
epochs = 100
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
auc_his = []

loss_val_his = []
sv_val_his = []
vae_val_his = []
auc_val_his =[]

min_loss = 9e10

for each_epoch in range(epochs):
  for each_batch in range(0, len(train_ehr), batch_size):
    batch_x, batch_len, batch_type, batch_y = get_batch(each_batch, batch_size, train_ehr, train_label, train_type, device)
    optimizer.zero_grad()
    lstm.train()
    vae.train()
    c_model.train()
    
    ehr_representation = lstm(batch_x, batch_len, device)
    loss, sv_loss, lpx_z, kld, inc_kld, kld2, pred, true = vae_objectives.objective(vae, c_model, device, x=ehr_representation, mask=batch_type, types=types, label=batch_y, beta=beta, nu=nu, K=1, kappa=1.0, components=True)
    mean_auc = roc_auc_score(true.flatten(), pred.flatten())
    loss.backward()
    optimizer.step()
    
    auc_his.append(mean_auc)
    loss_his.append(loss.cpu().detach().numpy())
    sv_his.append(sv_loss.cpu().detach().numpy())
    vae_his.append((lpx_z.cpu().detach().numpy(), kld.cpu().detach().numpy(), inc_kld.cpu().detach().numpy(), kld2.cpu().detach().numpy()))
    
    if each_batch % (batch_size*1) == 0:
      v_loss = - lpx_z + kld + inc_kld + kld2
      print('Epoch %d, Batch %d, AUC: %.4f, Loss: %.4f, C-Loss: %.4f, V-Loss: %.4f'%(each_epoch, each_batch, mean_auc, loss.cpu().detach().numpy(), sv_loss.mean().cpu().detach().numpy(), v_loss.cpu().detach().numpy()))
    
    if each_batch % (batch_size*100) == 0:
      with torch.no_grad():
        batch_x, batch_len, batch_type, batch_y = get_batch(each_batch, batch_size, valid_ehr, valid_label, valid_type, device)
        lstm.eval()
        vae.eval()
        c_model.eval()
        
        ehr_representation = lstm(batch_x, batch_len, device)
        loss, sv_loss, lpx_z, kld, inc_kld, kld2, pred, true = vae_objectives.objective(vae, c_model, device, x=ehr_representation, mask=batch_type, types=types, label=batch_y, beta=beta, nu=nu, K=1, kappa=1.0, components=True)
        mean_auc = roc_auc_score(true.flatten(), pred.flatten())
        
        auc_val_his.append(mean_auc)
        loss_val_his.append(loss.cpu().detach().numpy())
        sv_val_his.append(sv_loss.cpu().detach().numpy())
        vae_val_his.append((lpx_z.cpu().detach().numpy(), kld.cpu().detach().numpy(), inc_kld.cpu().detach().numpy(), kld2.cpu().detach().numpy()))
      
        v_loss = - lpx_z + kld + inc_kld + kld2
        print('Epoch %d, Batch %d, AUC: %.4f, Valid Loss: %.4f, C-Loss: %.4f, V-Loss: %.4f'%(each_epoch, each_batch, mean_auc, loss.cpu().detach().numpy(), sv_loss.mean().cpu().detach().numpy(), v_loss.cpu().detach().numpy()))
    
        if loss < min_loss:
          min_loss = loss
          state = {
                'lstm': lstm.state_dict(),
                'cmodel': c_model.state_dict(),
                'vae': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': (each_epoch,each_batch)
            }
          torch.save(state, './log/model')
          print('\n------------ Save best model ------------\n')
