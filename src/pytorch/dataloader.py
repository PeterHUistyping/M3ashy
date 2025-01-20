''' Neural Network Data Loader
    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import sys
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent))

# --- 3rd party ---
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset

# --- my module ---

class DataLoader():
    '''
        Loading, Batching, Shuffling the data
    '''
    def __init__(self, dataset, batch_size, shuffle=True):
        '''
            @param  dataset: torch.utils.data.Dataset
            @param  batch_size: int
            @param  shuffle: bool
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            random_permutation = np.random.permutation(len(dataset))
        else:
            random_permutation = np.arange(len(dataset))
        self.batches = [random_permutation[i:i + batch_size]
                   for i in range(0, len(dataset), batch_size)]
        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self): 
        if self.batch_idx >= len(self.batches):
            raise StopIteration
        else:
            batch = self.batches[self.batch_idx]
            self.batch_idx += 1
            return self.dataset[batch]
