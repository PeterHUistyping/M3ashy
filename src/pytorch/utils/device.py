''' Pytorch device utility.
    @section type
    - cpu
    - cuda
    - mps (MacOS)
    
    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
import torch

device = torch.device("cuda" if torch.cuda.is_available()
                      else torch.device("mps") if
                      torch.backends.mps.is_available()
                      else "cpu")
