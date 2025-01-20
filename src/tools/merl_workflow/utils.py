''' Utility functions for the MERL dataset workflow
    @section Functionalities
    - index 1D-3D_tuple
    - RGB_permute_BRDF
    - current_datetime
    
    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import sys
import os
import struct
from datetime import datetime
import numpy as np
import torch
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

THETA_H_SAMPLE_SIZE = 90
THETA_D_SAMPLE_SIZE = 90
PHI_D_SAMPLE_SIZE = 180


def index_1D_to_3D_tuple(index):
    '''
        Convert index from 1D int to 3D tuple
        (theta_h, theta_d, phi_d)
    '''
    global THETA_D_SAMPLE_SIZE, THETA_H_SAMPLE_SIZE , PHI_D_SAMPLE_SIZE
    THETA_D_SAMPLE_SIZE = 1 + int(89)
    THETA_H_SAMPLE_SIZE = 1 + int(89)
    PHI_D_SAMPLE_SIZE = 1 + int(179)

    theta_h = index // (THETA_D_SAMPLE_SIZE * PHI_D_SAMPLE_SIZE)
    theta_d = (index - theta_h * THETA_D_SAMPLE_SIZE *
               PHI_D_SAMPLE_SIZE) // PHI_D_SAMPLE_SIZE
    phi_d = index - theta_h * THETA_D_SAMPLE_SIZE * \
        PHI_D_SAMPLE_SIZE - theta_d * PHI_D_SAMPLE_SIZE
    return (theta_h, theta_d, phi_d)


def index_3D_tuple_to_1D(index_tuple):
    '''
        Convert index from 3D tuple to 1D int
    '''
    global THETA_D_SAMPLE_SIZE, THETA_H_SAMPLE_SIZE , PHI_D_SAMPLE_SIZE
    THETA_D_SAMPLE_SIZE = 1 + int(89)
    THETA_H_SAMPLE_SIZE = 1 + int(89)
    PHI_D_SAMPLE_SIZE = 1 + int(179)

    return index_tuple[0] * THETA_D_SAMPLE_SIZE * PHI_D_SAMPLE_SIZE + \
        index_tuple[1] * PHI_D_SAMPLE_SIZE + index_tuple[2]


def halfdiff2cartesian(theta_h, theta_d, phi_d):
    '''
        (theta_h, theta_d, phi_d, phi_h = 0) -> (hx, hy, hz, dx, dy, dz)
    '''
    # do both np and torch depending on the input type
    # theta_h
    if torch.is_tensor(theta_h):
        hx = torch.sin(theta_h) * np.cos(0.0)
        hy = torch.sin(theta_h) * np.sin(0.0)
        hz = torch.cos(theta_h)
        dx = torch.sin(theta_d) * torch.cos(phi_d)
        dy = torch.sin(theta_d) * torch.sin(phi_d)
        dz = torch.cos(theta_d)
    else:        
        hx = np.sin(theta_h) * np.cos(0.0)
        hy = np.sin(theta_h) * np.sin(0.0)
        hz = np.cos(theta_h)
        dx = np.sin(theta_d) * np.cos(phi_d)
        dy = np.sin(theta_d) * np.sin(phi_d)
        dz = np.cos(theta_d)
    return (hx, hy, hz, dx, dy, dz)


def RGB_permute_BRDF(brdf, file_index=1, verbose=False):
    '''
        Data Augmentation, permute the RGB dimension of the array (_, 3)
        @param brdf: array
        @return permuted brdf array
    '''
    if file_index < 0:
        return brdf
    else:
        rgb_type = file_index % 6
    if verbose:
        print("RGB permutation type {}".format(rgb_type))
    if rgb_type == 1:
        brdf = brdf[..., [0, 1, 2]]  # merl_1
    elif rgb_type == 2:
        brdf = brdf[..., [0, 2, 1]]  # merl_2
    elif rgb_type == 3:
        brdf = brdf[..., [1, 0, 2]]  # merl_3
    elif rgb_type == 4:
        brdf = brdf[..., [1, 2, 0]]  # merl_4
    elif rgb_type == 5:
        brdf = brdf[..., [2, 0, 1]]  # merl_5
    elif rgb_type == 0:
        brdf = brdf[..., [2, 1, 0]]  # merl_6
    return brdf


def current_datetime():
    '''
        @return string %m-%d_%H-%M-%S_
    '''
    date_time = datetime.now().strftime("%m-%d_%H-%M-%S_") # %Y
    return str(date_time)  