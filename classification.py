import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        #out = self.fc2(out)
        #out = torch.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out) 
        return out
      
class Classifications(nn.Module):
    def __init__(self, data_dim, hidden_dim, types):
        super(Classifications, self).__init__()
        self.mlps = nn.ModuleList([MLP(data_dim, hidden_dim) for i in range(types)])
    def forward(self, x, device):
        pred = []
        for m in self.mlps:
            pred.append(m(x))
        pred = torch.stack(pred).to(device) #k*B*1
        pred = pred.squeeze(-1).permute(1,0) #B*k
        return pred