import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    meta_learner import Learner
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, meta_lr, update_lr, update_step, data_dim, hidden_dim, types, init_vars, alpha):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.update_step = update_step
        self.alpha = alpha
        self.types = types

        config = [
        ('linear', [data_dim, hidden_dim]),
        ('relu', [True]),
        ('linear', [hidden_dim, 1]),
        ('sigmoid')
        ]

        self.common_net = nn.ModuleList([Learner(config) for i in range(types)])

        for i in range(types):
            common_net[i].init_vars(init_vars[i])

        self.target_net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [K, b, h]
        :param y_spt:   [K, b]
        :param x_qry:   [b, h]
        :param y_qry:   [b]
        :return:
        """
        bce = torch.nn.BCELoss()
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(self.types):
            # 1. run the i-th task and compute loss for k=0
            logits = self.common_net[i](x_spt[i], vars=None)
            loss = bce(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.common_net[i].parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.common_net[i].parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                logits_q = self.target_net(x_qry, self.target_net.parameters())
                loss_q = bce(logits_q, y_qry)
                losses_q[0] += self.alpha[i] * loss_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.target_net(x_qry, fast_weights)
                loss_q = bce(logits_q, y_qry)
                losses_q[1] += self.alpha[i] * loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.common_net[i](x_spt[i], fast_weights)
                loss = bce(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.target_net(x_qry, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = bce(logits_q, y_qry)
                losses_q[k + 1] += loss_q

            common_net[i].init_vars(fast_weights)

        loss_q = losses_q[-1] / self.types
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        return loss_q