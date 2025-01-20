''' Model factory class
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
sys.path.append(str(Path(__file__).parent.parent))
# --- 3rd party ---
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# --- related module ---
from pytorch.models.mlp import MLP
from pytorch.models.hyper_diffusion import HyperDiffusion
from pytorch.models.nbrdf import NBRDF
from pytorch.utils.device import device

# Dimension after data encoding
input_size = 6        # 3-dim or 6-dim
hidden_size = 21 * 1  # 21 * 3
output_size = 3

resolution_tuple = (101, 101)
resolution_size = resolution_tuple[0] * resolution_tuple[1]

# (1 + w) * Cond - w * unCond
# w = -1 (unCond) | w = 0 (Cond) | w > 0 (Guided Cond)
unconditional_guidance_scale = 0

class Trainer():
    '''
        Trainer config class
            A factory class creating new models with hyperparameters
        including model, batch_size, epochs, device, args
    '''
    def __init__(self, args=None, verbose=False):
        self.args = args
        self.device = device
        self.model = None
        self.batch_size = 32 * resolution_size
        self.epochs = 1
        self.optimizer = None
        self.scheduler = None            # learning rate scheduler
        self.stop_loss_threshold = 1e-3  # early stop diffusion training
        self.lr = 5e-3                   # diffusion adaptive learning rate
        self.verbose = verbose
        self.hyper = {
            "batch_size": self.batch_size, 
            "epochs": self.epochs,
            "device": self.device,
            "optimizer": self.optimizer,
        }

        self.mlp_settings = {
                "model": "MLP",
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
            }
        
        self.hyperdiffusion_settings = {
                "model": "HyperDiffusion",
                "use_cond": False,
                "cond_label":{
                    "unconditional_guidance_scale": unconditional_guidance_scale,
                }
            }
        
    def print_json(self, data):
        print(json.dumps(data, indent=4))

    def set_model_hyper_diffusion(self):
        '''
            Update model to HyperDiffusion if not already
        '''
        self.epochs = 1   # 200
        self.lr = 5e-4      # (default:5e-4) 5e-5, 5e-6
        self.stop_loss_threshold = 0.1     # 0.1, 5e-3, 1e-3

        if not isinstance(self.model, HyperDiffusion) or \
        (isinstance(self.model, HyperDiffusion)):
            self.model = HyperDiffusion(unconditional_guidance_scale)
            self.batch_size = 16  # 16, 25, 32
            self.model.to(device)
            
        self.print_json(self.hyperdiffusion_settings)
        self.save_hyperparameters()
        return self.model
    
    def set_model_MLP(self):
        '''
            Update model to MLP if not already
        '''
        if not isinstance(self.model, MLP):
            self.model = MLP(input_size, hidden_size, output_size)
            # self.model = NBRDF()

            # self.batch_size = 32 * resolution_size  // 4
            self.batch_size = 512

            self.model.to(device)
            # print("Model used: MLP")
            if self.verbose:
                self.print_json(self.mlp_settings)
            self.save_hyperparameters()
        return self.model
    
    
    def get_batch_size(self):
        print("Batch size: ", self.batch_size)
        return self.batch_size
    
    def get_num_epochs(self):
        print("Number of epochs: ", self.epochs)
        return self.epochs

    def get_model(self):
        return self.model

    def get_lr(self):
        return self.lr
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def ini_optimizer(self, model=None, lr=5e-3):
        '''
            Initialize optimizer for the model if not already
        '''
        if model == None:
            model = self.model

        if self.optimizer == None or lr != self.optimizer.param_groups[0]['lr']:
            self.optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': lr}],
                           lr=lr,
                           eps=1e-15,
                           weight_decay=0  # 1e-5
                           )
    
    def get_optimizer(self, model=None):
        '''
            Get optimizer for the model if not already
        '''
        return self.optimizer
    
    def get_scheduler(self, optimizer=None):
        if optimizer == None:
            optimizer = self.optimizer
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=30, 
                gamma=1, # 0.1
                last_epoch=80)

        return self.scheduler
    
    def get_loss_threshold(self):
        return self.stop_loss_threshold
    
    def load_model(self, path):
        if self.model == None:
            self.model = self.set_model_MLP()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self.model

    def save_checkpoint(self, path, loss, epoch=None):   
        '''
            Saving lr_scheduler is optional
        '''
        torch.save({
            'epoch': self.epochs if epoch == None else epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ini_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if self.scheduler != None:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
        self.model.eval()
        print(f"Resume the model from CHECKPOINT epoch = {epoch}.")
        return self.model, epoch, loss
    
    def get_hyperparameters(self):
        '''
            Update and return hyperparameters
        '''
        self.hyper = {
            "batch_size": self.batch_size, 
            "epochs": self.epochs,
            "device": self.device
        }
        return self.hyper
    
    def save_hyperparameters(self):
        self.get_hyperparameters()
        if self.verbose:
            print(self.hyper)
        # TODO: save to file "opt.txt"
        return self.hyper

    def get_args(self):
        return self.args

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
