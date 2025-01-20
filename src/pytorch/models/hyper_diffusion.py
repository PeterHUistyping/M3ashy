''' HyperDiffusion model
    @section Parameters
    - transformer-based
    
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
from .unet import Unet
from .diffusion.gaussian_diffusion import (GaussianDiffusion, LossType,
                                          ModelMeanType, ModelVarType)
from .transformer import Transformer
from ..utils.device import device

class HyperDiffusion(nn.Module):
    '''Pytorch HyperDiffusion model'''
    def __init__(self, unconditional_guidance_scale = 1.0):
        super().__init__()

        self.normalization_factor = 1.0
        self.unconditional_guidance_scale = unconditional_guidance_scale
        time_steps = 100
        betas = torch.tensor(np.linspace(1e-4, 2e-2, time_steps))
        self.image_size =  (10, 675)    # 612 675
        self.mlp_kwargs = None  # TODO: add mlp_kwargs
        
        # self.model = Unet(in_channels=1)
        self.model = Transformer(
            # Conditional Embedding dim 
            [675], ["mlp_weights"], # 612 675
            # layers, 
            # layer_names, 
            # **Config.config["transformer_config"]["params"]
        ).to(device)

        # Initialize diffusion utilities
        self.diff = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
            # diff_pl_module=self,
            unconditional_guidance_scale=self.unconditional_guidance_scale
        )

    def forward(self, x, y=None):
        t = (
            torch.randint(0, high=self.diff.num_timesteps, size=(x.shape[0],))
            .long()
            .to(device)
        )
        x = x * self.normalization_factor
        # sample from q(x_t | x_0)
        x_t, e = self.diff.q_sample(x, t)
        x_t = x_t.float()
        e = e.float()
        return self.model(x_t, t, y), e

     
   