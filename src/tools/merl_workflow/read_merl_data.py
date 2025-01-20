''' Read MERL data from binary file
    @section Input 
    - merl_data
    - merl_interpolated
    @section Format
    - tuple 
    - flatten array

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

from src.tools.merl_workflow.read_merl_list import split_linear_interpolation_str

output_folder_path = 'output/merl/'
merl_binary_name = 'blue-metallic-paint'

DATA_SIZE = 90 * 90 * 180


def transform_merl_array(merl_input_brdf):
    '''
        @param brdf in tuple
            shape: (3, 90, 90, 180)
            total size: 4374000
            format:
            array(0: color, 1: theta_h, 2: theta_d, 3: phi_d)

        @return transformed brdf in tuple
            shape: (90, 90, 180, 3)
            format:
            array(1: theta_h, 2: theta_d, 3: phi_d, 4: color)

        One by one implementation,

            MeasuredBRDF = np.array(merl_input.brdf).reshape(3, 90, 90, 180)
            # original index sequence as in input merl array
            # 0 1 2 3

            MeasuredBRDF = np.swapaxes(MeasuredBRDF, 0, 3)  # or np.moveaxes
            # after transformation
            # 3 1 2 0

            MeasuredBRDF = np.swapaxes(MeasuredBRDF, 0, 2)
            # after transformation
            # 2 1 3 0

            MeasuredBRDF = np.swapaxes(MeasuredBRDF, 0, 1)
            # after transformation
            # 1 2 3 0
    '''
    # or in one go,
    MeasuredBRDF = np.array(merl_input_brdf).reshape(3, 90, 90, 180)
    MeasuredBRDF = np.transpose(MeasuredBRDF, (1, 2, 3, 0))
    return MeasuredBRDF


# read from merl
def read_merl_data(filepath):
    '''
        Read MERL data from binary file
        # Format: (theta_h, theta_d, phi_d, color)
    '''
    merl_input = merl.Merl(filepath)
    return merl_input


def read_merl_data_tuple(file_path):
    '''
       @return Numpy array
            Shape: (90, 90, 180, 3)
    '''
    merl_input = read_merl_data(file_path)
    merl_data_tuple = transform_merl_array(merl_input.brdf)
    return merl_data_tuple


def read_merl_data_flatten(file_path):
    '''
       @return Numpy array
            Shape: (DATA_SIZE = 90*90*180, 3)
    '''
    merl_data_tuple = read_merl_data_tuple(file_path)
    merl_data_flatten = merl_data_tuple.reshape((DATA_SIZE, 3))
    return merl_data_flatten


def read_merl_interpolated_flatten(merl_brdf_name, input_folder_path):
    '''
        Interpolate BRDF data from the binary file
        @param str: BRDF_filename1_BRDF_filename2_w
        @return Numpy array
            Shape: (DATA_SIZE = 90*90*180, 3)
    '''
    merl_brdf_name_1, merl_brdf_name_2, weights = split_linear_interpolation_str(merl_brdf_name)
    merl_data_flatten_1 = read_merl_data_flatten(
    input_folder_path + merl_brdf_name_1 + '.binary')
    merl_data_flatten_2 = read_merl_data_flatten(
    input_folder_path + merl_brdf_name_2 + '.binary')
    merl_data_interpolated_flatten = np.full((DATA_SIZE,3), [-1,-1,-1], dtype=np.float32)
    index1 = np.arange(DATA_SIZE)
    index2 = np.arange(DATA_SIZE)

    index1_valid = np.array([i for i in index1 if np.all(merl_data_flatten_1[i,:] >= 0) ]) 
    index2_valid = np.array([i for i in index2 if np.all(merl_data_flatten_2[i,:] >= 0) ]) 
    index_valid = np.array([i for i in index2_valid if np.all(merl_data_flatten_2[i,:] >= 0) ]) 

    merl_data_interpolated_flatten[index_valid] = float(weights) * merl_data_flatten_1[index_valid] + (1-float(weights)) * merl_data_flatten_2[index_valid]
    return merl_data_interpolated_flatten


def read_merl_interpolated_tuple(merl_brdf_name, input_folder_path):
    '''
        Interpolate BRDF data from the binary file
        @param str: BRDF_filename1_BRDF_filename2_w
        @return Numpy array
            Shape: (90, 90, 180, 3)
    '''
    merl_data_interpolated_flatten = read_merl_interpolated_flatten(merl_brdf_name, input_folder_path)
    merl_data_interpolated_tuple = merl_data_interpolated_flatten.reshape((90, 90, 180, 3))
    return merl_data_interpolated_tuple