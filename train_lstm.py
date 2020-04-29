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

from naive_lstm import LSTM
import importlib

train_ehr1 = pickle.load(open('./dataset/train_dataset1', 'rb'))
train_ehr2 = pickle.load(open('./dataset/train_dataset2', 'rb'))
train_ehr3 = pickle.load(open('./dataset/train_dataset3', 'rb'))
train_ehr = train_ehr1+train_ehr2+train_ehr3
train_type = pickle.load(open('./dataset/train_type', 'rb'))
train_label = pickle.load(open('./dataset/train_scoring', 'rb'))

valid_ehr = pickle.load(open('./dataset/valid_dataset', 'rb'))
valid_type = pickle.load(open('./dataset/valid_type', 'rb'))
valid_label = pickle.load(open('./dataset/valid_label', 'rb'))

def get_batch(idx, batch_size, x, y, types, selected_type, device):
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
  batch_len = []
  batch_y = []
  for i in range(idx, end):
    batch_x[i-idx, :len(x[i]), :] = x[i]
    batch_len.append(len(x[i]))
    if selected_type in types[i]:
      batch_y.append(1)
    else:
      batch_y.append(0)

  batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
  batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
  batch_len = torch.tensor(batch_len, dtype=torch.long).to(device)
  
  return batch_x, batch_len, batch_y

batch_size = 128
epochs = 50
lr=1e-3

data_dim = 1823
embedding_dim = 512
hidden_dim = 320
device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')

lstm = LSTM(data_dim=data_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(list(lstm.parameters()), lr=lr)

min_loss = 9e10

for each_epoch in range(epochs):
  for each_batch in range(0, len(train_ehr), batch_size):
    batch_x, batch_len, batch_y = get_batch(each_batch, batch_size, train_ehr, train_label, train_type, 1, device)
    optimizer.zero_grad()
    lstm.train()
    
    output = lstm(batch_x, batch_len, device)
    bce = torch.nn.BCELoss()
    loss = bce(output, batch_y.unsqueeze(-1))
    loss.backward() 
    optimizer.step()
    
    if each_batch % (batch_size*1) == 0:
      if np.sum(batch_y.cpu().detach().numpy()) > 0:
        auroc = roc_auc_score(batch_y.cpu().detach().numpy(), output.cpu().detach().numpy())
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(batch_y.cpu().detach().numpy(), output.cpu().detach().numpy())
        auprc = metrics.auc(recalls, precisions)
      else:
        auroc = -1
      print('Epoch %d, Batch %d, Loss: %.4f, ROC: %.4f, PRC: %.4f'%(each_epoch, each_batch, loss.cpu().detach().numpy(), auroc, auprc))
      
    if each_batch % (batch_size*100) == 0 and each_batch != 0:
      with torch.no_grad():
        lstm.eval()
        tmp_loss = []
        pred_his = []
        true_his = []
        for valid_batch in range(0, len(valid_ehr), batch_size):
          batch_x, batch_len, batch_y = get_batch(valid_batch, batch_size, valid_ehr, valid_label, valid_type, 1, device)

          output = lstm(batch_x, batch_len, device)
          bce = torch.nn.BCELoss()
          loss = bce(output, batch_y.unsqueeze(-1))
          
          pred_his+=list(output.cpu().detach().numpy())
          true_his+=list(batch_y.cpu().detach().numpy())
          tmp_loss.append(loss.cpu().detach().numpy())
        
          auroc = roc_auc_score(true_his, pred_his)
          (precisions, recalls, thresholds) = metrics.precision_recall_curve(true_his, pred_his)
          auprc = metrics.auc(recalls, precisions)
          print('Epoch %d, Batch %d, Valid Loss: %.4f, ROC: %.4f, PRC: %.4f'%(each_epoch, valid_batch, tmp_loss[-1], auroc, auprc))