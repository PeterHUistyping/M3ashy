'''  
    NBRDF MLP model (Pytorch)
    
    Input: Cartesian coordinate for positional samples 
            (1: theta_h, 2: theta_d, 3: phi_d, 4: phi_h = 0) -> (hx, hy, hz, dx, dy, dz)
    Output: MERL reflectance value
    
    - input_size   6
    - hidden_size  21
    - hidden_layer 3
    - output_size  3
    
    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import math
import gc
from datetime import datetime

import sys
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent))
# --- 3rd party ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# --- related module ---
from pytorch.models.encoder.hashGridEncoding import MultiResHashGrid
from pytorch.models.encoder.fourierEncoding import FourierEncoding
from pytorch.utils.device import device


class MLP(nn.Module):
    '''Pytorch MLP model'''
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        # self.pos_emb1 = FourierEncoding(in_channels=3)
        # self.pos_emb2 = MultiResHashGrid(dim=3)
        '''
        self.f = nn.Sequential(
            # fc1
            nn.Linear(input_size, hidden_size, bias=True),
            # nn.BatchNorm1d(hidden_size),

            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

            # fc2
            nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.BatchNorm1d(hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

            # output_layer
            nn.Linear(hidden_size, output_size, bias=True),
            # nn.Sigmoid()
        )
        '''

        # Initialize separately
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=True)
        # initialize the weight

        # Reproducibility for generation purpose
        torch.manual_seed(0)
        random.seed(0)
        with torch.no_grad():
            for func in [self.fc1, self.fc2, self.fc3]:
                func.bias.zero_()
                func.weight.uniform_(0.0, 0.02)

    def forward(self, x):
        # out = self.pos_emb1(x)
        # out = self.pos_emb2(x)
        # out = F.relu(torch.exp(self.f(x)) - 1.0)
        
        out = self.fc1(x)
        out = F.leaky_relu(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)
        out = self.fc3(out)
        out = F.relu(torch.exp(out) - 1.0)
        # If there's invalid data, the output will be negative
        # out = torch.exp(self.f(x)) - 1.0

        return out
