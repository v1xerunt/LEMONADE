import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, data_dim, embedding_dim, hidden_dim):
      super(LSTM, self).__init__()
      self.embedding_dim = embedding_dim
      self.hidden_dim = hidden_dim

      self.nn_embd = nn.Linear(data_dim, embedding_dim)
      self.net=nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x, mask, device):
      batch_size, time_step, feature_dim = x.size()
      h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
      c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)

      x = x.contiguous().view(-1, x.size(-1))
      x = self.nn_embd(x)
      x = x.contiguous().view(batch_size, time_step, self.embedding_dim)
      out, _ = self.net(x, (h0, c0))
      mask = (mask-1).unsqueeze(1).unsqueeze(2).expand(out.size(0), time_step, out.size(2))
      out = out.gather(1, mask)[:,0]
      return out