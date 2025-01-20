''' NBRDF weights dataset for PyTorch
    @section Reference
    - Pytorch tutorial 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import sys
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# --- 3rd party ---
import numpy as np
import pandas as pd
import src.tools.merl_workflow.lib.merl as merl
import torch
from torch.utils.data import Dataset

# --- my module ---
from src.pytorch.utils.device import device

class NBDRFWeightsDataset(Dataset):
    '''
        MLP weights for each BRDF 
            training data: split_train_pct=90% 
            inference data: (1-split_train_pct)=10%

        file_index <= 6: RGB permuted data
            @param  type == 1: ALL data
                    type == 2: train data
                    type == 3: infer data
    '''
    def __init__(self, 
                 split_train_pct=0.95, 
                 type=1,
                 mlp_weights_path = "data/NeuMERL/",
                 load_all_files=True,
                 max_files=25): # 7, 25
        
        self.type = type
        self.split_train_pct = split_train_pct
        # numpy_weights = np.load(mlp_weights_path + "mlp_weights_all.npy")

        # load [1,Max-1]
        if load_all_files:
            numpy_weights = np.load(mlp_weights_path + "NeuMERL-2400.npy")
        else:
            numpy_weights = np.load(mlp_weights_path + "mlp_weights_all_1.npy")
            for i in range(2, max_files): 
                # print("Read RGB 6* + Interpolate")
                numpy_weights2 = np.load(mlp_weights_path + "mlp_weights_all_" + str(i) + ".npy")
                numpy_weights = np.vstack((numpy_weights, numpy_weights2))  
        print(numpy_weights.shape)

        if numpy_weights.ndim == 1:
            numpy_weights = numpy_weights.reshape(1, -1)

        self.mlp_weights = torch.from_numpy(numpy_weights)
        
        self.index_train, self.index_infer = self.data_split(split_train_pct)

    def __len__(self):
        if self.type == 1:
            length = len(self.mlp_weights)
        elif self.type == 2:
            length = len(self.index_train)
        elif self.type == 3:
            length = len(self.index_infer)
        return length
    
    def __getitem__(self, idx):
        '''
            Loads and returns a sample from the dataset at the given index idx
            @return mlp_weights: tensor to device, 
                    label: type undetermined
        '''
        if self.type == 1:
            idx = idx
        elif self.type == 2:
            idx = self.index_train[idx]
        elif self.type == 3:
            idx = self.index_infer[idx]
        
        return self.mlp_weights[idx].float().to(device), None
    
    def update_type(self, type):
        '''
            Update the type of the dataset

            @param  type == 1: ALL data
                    type == 2: train data
                    type == 3: infer data
        '''
        self.type = type

    def data_split(self, split_train_pct=0.95):
        '''
            Split the dataset into 80% training and 20% inference
            with data preprocessing removing invalid data points.

            (1: theta_h, 2: theta_d, 3: phi_d, 4: color)
            Notice that the representation is 3D, rather than 6D.
        '''
        if True:
            seed_value = 0
            np.random.seed(seed_value)
            print(f"Train/Infer split with seed value: {seed_value}")
        index0 = np.random.permutation(len(self.mlp_weights))
      
        TRAIN_SIZE = int(split_train_pct * len(self.mlp_weights))
        print(f"Train size: {TRAIN_SIZE} ", f"Infer size: {len(self.mlp_weights) - TRAIN_SIZE}")

        index_train = index0[0: TRAIN_SIZE]
        index_infer = index0[TRAIN_SIZE: ]

        # write index_infer to file
        if False:
            print(f"Write index_infer array to file")
            np.save(f"output/generation/index_infer_seed{seed_value}_pct{self.split_train_pct}.npy", index_infer)

        return index_train, index_infer
    
if __name__ == '__main__':
    nbrdf_weights_dataset = NBDRFWeightsDataset()
    print(nbrdf_weights_dataset[0])
