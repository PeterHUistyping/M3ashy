''' Measured BRDF dataset for PyTorch
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
import torch
from torch.utils.data import Dataset
import src.tools.merl_workflow.lib.merl as merl
# --- my module ---
from src.pytorch.utils.device import device
from src.tools.merl_workflow.read_merl_data import read_merl_data_flatten, read_merl_data_tuple, read_merl_interpolated_flatten
from src.tools.merl_workflow.utils import index_1D_to_3D_tuple, RGB_permute_BRDF, halfdiff2cartesian

DATA_SIZE = 90 * 90 * 180

class MERLDataset(Dataset):
    '''
        MERL BRDF data 
            training data: split_train_pct=80% 
            inference data: (1-split_train_pct)=20%
    '''
    def __init__(self, merl_folder_path, merl_binary_name, 
                 split_train_pct=0.8, 
                 file_index=1, type=1):
        '''
            @param  file_index >  6: index of the interpolated file
                    file_index == 1: use the original data
                    file_index <= 6: RGB permuted data
            @param  type == 1: ALL data
                    type == 2: train data
                    type == 3: infer data
        '''
        self.type = type
        if file_index <= 6:
            merl_data = read_merl_data_flatten(merl_folder_path + merl_binary_name + ".binary")
            # Data Augmentation
        else:
            merl_data = read_merl_interpolated_flatten(merl_binary_name, merl_folder_path)
        merl_data = RGB_permute_BRDF(merl_data, file_index)
        
        self.merl_data = merl_data.astype(np.float32)
        self.index_train, self.index_infer = self.data_split(split_train_pct)

    def __len__(self):
        length = len(self.merl_data)
        if self.type == 1:
            length = len(self.merl_data)
        elif self.type == 2:
            length = len(self.index_train)
        elif self.type == 3:
            length = len(self.index_infer)
        return length
    
    def __getitem__(self, idx):
        '''
            Loads and returns a sample from the dataset at the given index idx

            @return X_pos_torch: tensor to device,
                    Y_brdf_torch: tensor to device
        '''
        if self.type == 1:
            idx = idx
        elif self.type == 2:
            idx = self.index_train[idx]
        elif self.type == 3:
            idx = self.index_infer[idx]

        X_pos = index_1D_to_3D_tuple(idx)
        # Used for 6-dim input
        X_pos = halfdiff2cartesian(
            np.square(X_pos[0]  / 90.0) * (np.pi/2), 
            X_pos[1] / 90.0 * (np.pi/2), 
            X_pos[2] / 180.0 * np.pi)
        
        X_pos = np.asarray(X_pos).T.astype(np.float32)
        # convert to cartesian coordinates
        Y_brdf = self.merl_data[idx]
        # transform to tensor and proper device
        X_pos_torch = torch.tensor(X_pos).to(device)
        Y_brdf_torch = torch.from_numpy(Y_brdf).to(device)
        return X_pos_torch, Y_brdf_torch
    
    def update_type(self, type):
        '''
            Update the type of the dataset

            @param  type == 1: ALL data
                    type == 2: train data
                    type == 3: infer data
        '''
        self.type = type
       
    def data_split(self, split_train_pct):
        '''
            Split the dataset into 80% training and 20% inference
            with data preprocessing removing invalid data points.

            (1: theta_h, 2: theta_d, 3: phi_d, 4: color)
            Notice that the representation is 3D, rather than 6D.
        '''
        index0 = np.random.permutation(DATA_SIZE)
        # check when the value is (-1, -1, -1)

        # Data preprocessing
        index0_invalid = np.array([i for i in index0 if np.any(self.merl_data[i,:] <0)]) 
        index0_valid = np.array([i for i in index0 if np.all(self.merl_data[i,:] >= 0)])
        
        # assert(len(index0_invalid) + len(index0_valid) == DATA_SIZE)
    
        TRAIN_SIZE = int(split_train_pct * len(index0_valid))
        
        index_train = index0_valid[0: TRAIN_SIZE]
        index_infer = index0_valid[TRAIN_SIZE: ]

        return index_train, index_infer

if __name__ == '__main__':
    merl_dataset = MERLDataset("data/merl/", "blue-metallic-paint")
    print(merl_dataset[0])
