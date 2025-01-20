''' Read merl related texts from file
    @section Input
    - merl_dataset_list
    - interpolation
    
    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file
'''
# --- built in ---
import sys
import os
import random

# --- 3rd party ---
import torch
import numpy as np
import torch.nn as nn
 

def read_text_from_file(file_path="data/merl/merl_dataset_list.txt"):
    merl_dataset_list = []
    with open(file_path, 'r') as f:
        # readline
        for line in f:

            if line == "":
                break

            # strip '\n'
            merl_dataset_list.append(line.strip())
    f.close()
    return merl_dataset_list


def gen_merl_linear_interpolation(max_size = 100, file_index = 0):
    '''
        Generate a pair of BRDF and their weights (w, 1-w)
    '''
    merl_text_label = read_text_from_file()

    with open("data/merl/interpolate/interpolation_"+ str(file_index) +".txt", 'w') as f:
        for i in range(0, max_size):
            ind1, ind2 = random.sample(range(0, len(merl_text_label)), 2)
            w = random.uniform(0.3, 0.7)
            w_display = round(w, 6)
            # print(f"Interpolating {merl_text_label[ind1]} and {merl_text_label[ind2]} with first BRDF weight {w_display}")
            f.write(f"{merl_text_label[ind1]} {merl_text_label[ind2]} {w_display}\n")


def gen_merl_linear_interpolation_batch(max_batch_size = 50):
    '''
        Generate 50 batches of linear interpolated BRDF data.
    '''
    for i in range(0, max_batch_size):
        gen_merl_linear_interpolation(max_size = 100, file_index=i)


def read_merl_linear_interpolation(file_index = 0):
    '''
        @return list: 100 interpolated BRDF pairs information
            format: [BRDF_filename1_BRDF_filename2_w, ...]
    '''
    merl_pair_list = []
    with open("data/merl/interpolate/interpolation_"+ str(file_index) +".txt", 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            # strip '\n'
            line = line.strip()
            chunks = line.split(' ')
            assert(len(chunks)==3)
            merl_pair_list.append(chunks[0]+"_"+chunks[1]+"_"+chunks[2])

            # merl_interpolated_pair = line.split()
            # split by space

            # print(merl_interpolated_pair)
            # BRDF_filename1 = merl_interpolated_pair[0]
            # BRDF_filename2 = merl_interpolated_pair[1]
            # w_display = merl_interpolated_pair[2]
            # print(f"Interpolating {BRDF_filename1} and {BRDF_filename2} with first BRDF weight {w_display}")
    print(len(merl_pair_list))
    assert(len(merl_pair_list) == 100)
    return merl_pair_list


def split_linear_interpolation_str(str):
    ''' 
        @param str: BRDF_filename1_BRDF_filename2_w
        @return BRDF_filename1, BRDF_filename2, w
    '''
    # if the input is type of string
    chunks = str.split('_')

    return chunks


def read_merl_mixed_list(file_index, verbose=True):
    '''
        @param  file_index: int, 0-MAX
        @return file_index >  6: the interpolated BRDF pairs
                file_index <= 6: original merl dataset list 
            RGB permutation rule is related to (file_index mod 6)
    '''
    if file_index <= 6:
        return read_text_from_file()
    else:
        interpolate_index = (file_index-7)//6
        if verbose:
            print(f'Interpolate_index {interpolate_index}')
        return read_merl_linear_interpolation(interpolate_index)


if __name__ == "__main__":
    # gen_merl_linear_interpolation_batch()   
    # Warning: overwriting the original text file
    
    read_merl_linear_interpolation(file_index = 0)
 