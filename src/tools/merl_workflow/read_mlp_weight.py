''' Read mlp weights from model weights file
    @section Input
    - Supervised NBRDF
    - Synthetic NBRDF

    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import sys
import argparse
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# --- 3rd party ---
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# --- my module ---
# import src.pytorch.train as train
# from src.pytorch.train import model, trainer
from src.tools.merl_workflow.read_merl_list import read_merl_mixed_list
from src.pytorch.model_factory import Trainer

# read from merl

# weight_name = ['f.0.weight', 'f.0.bias', 'f.3.weight',
#                'f.3.bias', 'f.6.weight', 'f.6.bias']

weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight',
               'fc2.bias', 'fc3.weight', 'fc3.bias']

input_folder_path = 'data/merl/'
output_folder_path = 'output/merl/'
generation_folder_path = 'output/generation/mlp/'
merl_binary_name = 'blue-metallic-paint'
file_index = 1

trainer = Trainer()
model = trainer.set_model_MLP()


def load_mlp_model(filepath):
    # transform_merl_array
    # train.model = train.trainer.set_model_MLP()
    # return train.trainer.load_model(filepath)
    global trainer
    return trainer.load_model(filepath)


def load_mlp_weight_dict(filepath):
    # transform_merl_array
    model = load_mlp_model(filepath)
    return model.state_dict()


def load_mlp_weight(filepath):
    '''
        @param filepath: str

        @return mlp_weight: 1D concatenated np.array
        mlp_weight_shape: 1D np.array
    '''
    # transform_merl_array
    mlp_weight = np.array([], dtype=np.float32)
    mlp_weight_shape = np.array([[]], dtype=np.int32)
    state_dict = load_mlp_weight_dict(filepath)

    for i, weight_name_i in enumerate(weight_name):

        np_array = state_dict[weight_name_i].detach().cpu().numpy()

        np_array_shape = np.array(np_array.shape, dtype=np.int32)

        # print(weight_name_i, "\t", np_array.shape)

        np_array = np_array.reshape(-1)

        # print(np_array)

        mlp_weight = np.concatenate((mlp_weight, np_array), axis=0)

    # print("mlp_weight", mlp_weight.shape)
    # print(np.min(mlp_weight), np.max(mlp_weight))
    return mlp_weight


def change_mlp_weight(index, mlp_weight):
    '''
        fc1.weight       (21, 3) or (21, 6)
        fc1.bias         (21,)
        fc2.weight       (21, 21)
        fc2.bias         (21,)
        fc3.weight       (3, 21)
        fc3.bias         (3,)

        mlp_weight       (612,) or (675,)
        save the mlp weight as specified in the mlp_weight

        return the size of the remaining mlp_weight,
            if the size is 0, then all the mlp_weight has been used
    '''
    global trainer
    model = trainer.set_model_MLP()

    filepath = output_folder_path + merl_binary_name + "/model.pth"
    model = load_mlp_model(filepath)
    # for i, param in enumerate(model.parameters()):
    #     print("i", i, param.size())
    
    skip_weights = -1
    # position encoding
    # skip_weights = 15

    # Without position encoding
    # skip_weights = 5
    for i, param in enumerate(model.parameters()):

        if i > skip_weights:
            # print(type(param), param.size())
            shape = param.size()
            total_size = np.prod(shape)
            if len(shape) == 1:
                param.data = torch.from_numpy(
                    mlp_weight[:total_size]).reshape(shape)
            else:
                param.data = torch.from_numpy(
                    mlp_weight[:total_size].reshape(shape[0], shape[1]))
            mlp_weight = mlp_weight[total_size:]
    assert (mlp_weight.size == 0)
    # train.save_model("output/generation" +  "/model.pth")
    # train.trainer.save_model(output_folder_path + merl_binary_name + "/model"+ str(index) +".pth")

    trainer.save_model(
        generation_folder_path + "mlp_gen" + str(index) + "/"
        "model" +
        str(index) +
        ".pth")

    print(
        "Saved model: " +
        generation_folder_path + "mlp_gen" + str(index) + "/"
        "model" +
        str(index) +
        ".pth")

    return mlp_weight.size


def read_new_mlp_weights(filepath):
    '''
        read numpy array mlp_weights from file

        @return mlp_weights_all: np.array 2D
    '''
    # read numpy array mlp_weight from file
    mlp_weights_all = np.load(filepath)
    if mlp_weights_all.ndim == 1:
            mlp_weights_all = mlp_weights_all.reshape(1, -1)
    return mlp_weights_all


def write_merl_full_mlp_weights(merl_data_list, file_index):
    # write numpy array mlp_weight to file and read it later

    mlp_weights_all = np.array([], dtype=np.float32)
    first_valid = True
    for i, binary_name in enumerate(merl_data_list):
        # if i > 91:
        #     break
        # if "metallic" not in binary_name:
        #     continue
        mlp_weight = load_mlp_weight(
            output_folder_path + "merl_"+str(file_index)+"/" + binary_name + "/model.pth")
        if not first_valid:
            mlp_weights_all = np.row_stack((mlp_weights_all, mlp_weight))
        else:
            mlp_weights_all = mlp_weight
            first_valid = False
    print(mlp_weights_all.shape)
    np.save(f"output/generation/mlp_weights_all_{file_index}", mlp_weights_all)


def gen_merl_full_mlp_weights(file_index):
    '''
        read all the mlp weights from the merl dataset and save it to the mlp_weights_all.npy

        @return the size of the saved shape of numpy array in mlp_weights_all.npy
    '''
    merl_data_list = read_merl_mixed_list(file_index)   # read_text_from_file()
    write_merl_full_mlp_weights(merl_data_list, file_index)
    return read_new_mlp_weights(
        f"output/generation/mlp_weights_all_{file_index}.npy").shape


def gen_new_mlp_weights():
    '''
        Read in the mlp_weights_new.npy and saved the mlp weights to the model.pth
    '''
    mlp_weights_new = read_new_mlp_weights(
        "output/generation/" + "mlp_weights_new.npy")
    results = []
    for i in range(mlp_weights_new.shape[0]):
        results.append(change_mlp_weight(i, mlp_weights_new[i]))
    return results


if __name__ == "__main__":
    # train.pytorchModel = train.PytorchModel.MLP
    parser = argparse.ArgumentParser(description='Plot mosaic figure from rendered results.')
    parser.add_argument('-f', '--file_index', type=int, default=-1, help='file index to plot')
    args = parser.parse_args()
    file_index = args.file_index
    # print(f"file_index: {file_index}")
    
    # if len(sys.argv) >= 2:
    # file_index = int(sys.argv[1])

    if file_index > 0:
        gen_merl_full_mlp_weights(file_index)
    else:
        output_folder_path = output_folder_path + "merl_1/" 
        gen_new_mlp_weights()
