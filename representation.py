import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.net=nn.LSTM(data_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x, device):
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.net(x, (h0, c0)) 
        return out[:, -1, :]