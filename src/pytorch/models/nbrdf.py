import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MLP_(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features, exp_act_last_layer=True):
        super().__init__()
        self.exp_act_last_layer = exp_act_last_layer
        in_fc = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        middle_fcs = [nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=True) for _ in range(hidden_layers)]
        out_fc = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
        self.fcs = nn.ModuleList([in_fc] + middle_fcs + [out_fc])

        # initialize weights
        with torch.no_grad():
            for fc in self.fcs:
                fc.bias.zero_()
                fc.weight.uniform_(-0.05, 0.05)

    def forward(self, x):
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        if self.exp_act_last_layer:
            x = F.relu(torch.exp(self.fcs[-1](x)) - 1.0)
        else:
            x = self.fcs[-1](x)
        return x


class NBRDF(MLP_):
    def __init__(self):
        super().__init__(in_features=6, out_features=3, hidden_layers=1, hidden_features=21)
