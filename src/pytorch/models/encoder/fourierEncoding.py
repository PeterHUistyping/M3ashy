# --- built in ---
import math
import sys
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent))

# --- 3rd party ---
import numpy as np
import torch
import torch.nn as nn
# --- my module ---
from pytorch.utils.device import device


class FourierEncoding(nn.Module):
    def __init__(
            self,
            in_channels,
            N_freqs=12,
            logscale=None,
            mode='default_nrc') -> None:
        super(FourierEncoding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.mode = mode

        self.funcs = [torch.sin]
        # torch.cos

        self.out_channels = in_channels * (len(self.funcs) * N_freqs)
        if logscale:
            self.freq_bands = 2**torch.linspace(0,
                                                N_freqs - 1, N_freqs).to(device)
        else:
            self.freq_bands = torch.linspace(
                1, 2**(N_freqs - 1), N_freqs).to(device)

    def forward(self, x):
        '''
        embeds x to sin(2^k x)
        Inputs: x: (B, self.in_channels)
        Outputs: out: (B, self,out_channels)
        '''
        if self.mode == "full":
            out = [x]
        else:
            out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)
