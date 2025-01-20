import os

import numpy as np
import torch


# to scale the training data to [-1, 1] if range11 or [0, 1] if not range11
class MinMaxScaler:
    def __init__(self, range11=False, min=None, max=None):
        self.min = min
        self.max = max
        self.range11 = range11

    # Update the minimum and maximum values used for later scaling.
    # the min/max is computed over the last dimension
    def fit(self, data):  # Compute the minimum and maximum values used for later scaling.
        all_dim_except_last = tuple(range(data.dim() - 1))
        self.min = torch.amin(data, dim=all_dim_except_last)
        self.max = torch.amax(data, dim=all_dim_except_last)

    def scale(self, data):  # Scale the data
        if self.min is None or self.max is None:
            raise ValueError("You need to fit the scaler before transforming data!")

        # for single values, set it to the middle of the range
        single_value_inds = self.min == self.max
        data_return = data.clone()

        if self.range11:
            data_return[..., single_value_inds] = 0
            data_return[..., ~single_value_inds] = 2 * (data_return[..., ~single_value_inds] - self.min[~single_value_inds]) / (
                    self.max[~single_value_inds] - self.min[~single_value_inds]) - 1.0
        else:
            data_return[..., single_value_inds] = 0.5
            data_return[..., ~single_value_inds] = (data_return[:, ~single_value_inds] - self.min[~single_value_inds]) / (
                    self.max[~single_value_inds] - self.min[~single_value_inds])
        return data_return

    def descale(self, scaled_data):  # Transform the data back to its original space.
        if self.min is None or self.max is None:
            raise ValueError("You need to fit the scaler before inverting data!")
        if self.range11:
            return (scaled_data + 1.0) * (self.max - self.min) * 0.5 + self.min
        else:
            return scaled_data * (self.max - self.min) + self.min


def accumulate(target, source, decay=0.9999):  # https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
    source_para = dict(source.named_parameters())

    for k, v in target.named_parameters():
        v.data = v.data * decay + source_para[k].data * (1 - decay)


# flattens a state dict of a model to a vector
def model_to_flattened_weights(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])


# builds a model from a flattened vector
def flattened_weights_to_model(flattened_weights, model):
    pointer = 0
    with torch.no_grad():
        for param in model.parameters():
            num_param = param.numel()
            param_data = flattened_weights[pointer:pointer + num_param].view_as(param)
            pointer += num_param
            param.copy_(param_data)
    assert pointer == len(flattened_weights)
    return model


def readMERLBRDF(filename, scale=True):
    """Reads a MERL-type .binary file, containing a densely sampled BRDF

    Returns a 4-dimensional array (phi_d, theta_d, theta_h, channel)"""
    print("Loading MERL-BRDF: ", filename)
    try:
        f = open(filename, "rb")
        dims = np.fromfile(f, np.int32, 3)
        vals = np.fromfile(f, np.float64, -1)
        f.close()
    except IOError:
        print("Cannot read file:", os.path.basename(filename))
        return

    BRDFVals = np.swapaxes(np.reshape(vals, (dims[2], dims[1], dims[0], 3), 'F'), 1, 2)
    if scale:
        BRDFVals *= (1.00 / 1500, 1.15 / 1500, 1.66 / 1500)  # Colorscaling
    BRDFVals[BRDFVals < 0] = -1

    return BRDFVals


def saveMERLBRDF(filename, BRDFVals, shape=(180, 90, 90), scale=True):
    "Saves a BRDF to a MERL-type .binary file"
    print("Saving MERL-BRDF: ", filename)
    BRDFVals = np.array(BRDFVals)  # Make a copy
    assert BRDFVals.shape == (np.prod(shape), 3) or BRDFVals.shape == shape + (
        3,), f"Shape of BRDFVals incorrect: {BRDFVals.shape}, should be (180, 90, 90, 3) or (145800, 3)"

    if scale:
        BRDFVals /= (1.00 / 1500, 1.15 / 1500, 1.66 / 1500)  # Colorscaling

    # Are the values not mapped in a cube?
    if (BRDFVals.shape[1] == 3):
        BRDFVals = np.reshape(BRDFVals, shape + (3,))

    # Vectorize:
    vec = np.reshape(np.swapaxes(BRDFVals, 1, 2), (-1), 'F')
    shape = [shape[2], shape[1], shape[0]]

    try:
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", os.path.basename(filename))
        return


# compute the inner product along the last dimension
def inner_product(a, b, keepdim=False):
    return torch.sum(a * b, dim=-1, keepdim=keepdim)


if __name__ == "__main__":
    pass
