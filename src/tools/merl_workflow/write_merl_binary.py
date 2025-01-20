''' Covert from MLP model weights to MERL binary file
    @section Functionalities
    - MLP_to_MERL
    - MLP2_to_MERL_BATCH (reference)

    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import sys
import os
import struct
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# --- 3rd party ---
import numpy as np
import torch

# --- my module ---
from src.tools.merl_workflow.read_merl_data import read_merl_data_flatten
from src.tools.merl_workflow.utils import index_1D_to_3D_tuple, index_3D_tuple_to_1D, halfdiff2cartesian
from src.tools.merl_workflow.read_mlp_weight import load_mlp_model
from src.tools.merl_workflow.lib.utils import saveMERLBRDF, flattened_weights_to_model
from src.pytorch.utils.device import device

output_folder_path = 'output/merl/'
generation_folder_path = 'output/generation/mlp/'
merl_binary_name = 'blue-metallic-paint'


def write_merl_binary(BRDF_tuple, filepath):
    '''
        @param  merl BRDF tuple,
            shape: ('3i', 90, 90, 180)
            total size: 4374000
            format:
                array(0: color, 1: theta_h, 2: theta_d, 3: phi_d)
        @return None
    '''
    # write three integers to a binary file
    BRDF_tuple = BRDF_tuple.reshape(-1)
    print(BRDF_tuple.shape)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('3i', 90, 90, 180))
        f.write(struct.pack(str(3 * 90 * 90 * 180) + 'd', *BRDF_tuple))
    f.close()


def reverse_transform(BRDF):
    '''
        reverse the transformation in preprocess after reading MERL dataset
        @param BRDF_flatten or BRDF_tuple with shape (90, 90, 180, 3)
        @return BRDF_tuple with shape (3, 90, 90, 180)
    '''
    BRDF = BRDF.reshape((90, 90, 180, 3))
    BRDF_tuple = np.transpose(BRDF, (3, 0, 1, 2))
    return BRDF_tuple


def MLP_to_MERL(model):
    '''
        Convert MLP inference results to MERL lookup table
        @param model, pytorch model
        @return Generated_BRDF_tuple with shape (3, 90, 90, 180)
    '''
    # MLP inference
    # create 90*90*180 lookup table

    index_infer = np.arange(0, 90 * 90 * 180)
    infer_data_X = index_1D_to_3D_tuple(index_infer)
    # 6-dim
    id_theta_h = infer_data_X[0]
    id_theta_d = infer_data_X[1]
    id_phi_d = infer_data_X[2]
    infer_data_X = halfdiff2cartesian(np.square(id_theta_h/90.0)*(np.pi/2), id_theta_d/90.0*(np.pi/2), id_phi_d/180.0*np.pi)
    infer_data_X = np.asarray(infer_data_X)

    torch_infer_data_X = torch.tensor(
        infer_data_X.T, dtype=torch.float).to(device)
    MeasuredBRDF = model(torch_infer_data_X).detach().cpu().numpy()
    MeasuredBRDF = reverse_transform(MeasuredBRDF)
    return MeasuredBRDF.astype(np.float64)


def MLP_to_MERL_BATCH():
    for index in range(0, 100): # 100, 120
        model = load_mlp_model(generation_folder_path + "mlp_gen"
                               + str(index) + "/model" + str(index) + ".pth")
        Generated_BRDF = MLP_to_MERL(model)
        write_merl_binary(
            Generated_BRDF,
            generation_folder_path + "mlp_gen" + str(index) + 
            "/mlp_gen" + str(index) +
            '.binary')
        MeasuredBRDF_flatten = read_merl_data_flatten(
            generation_folder_path + "mlp_gen" + str(index) + 
            "/mlp_gen" + str(index) +
            '.binary')
     

if __name__ == "__main__":
    MLP_to_MERL_BATCH()
