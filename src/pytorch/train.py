''' Training script for Neural Network
    @section description Model factory
    - NBRDF using MLP model
    - Generation using HyperDiffusion model

    @author
    Copyright (c) 2024 - 2025 Peter HU.

    @file

'''
# --- built in ---
import gc
import sys
import argparse
from enum import Enum, IntEnum
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent))

# --- 3rd party ---
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
# isinstance(some_tensor, Float[Tensor, "batch channels height width"])
# from torch.utils.data import DataLoader
# --- my module ---
from src.pytorch.utils.device import device
from src.pytorch.dataloader import DataLoader
from src.tools.merl_workflow.read_merl_list import read_merl_mixed_list 
from src.tools.merl_workflow.read_mlp_weight import change_mlp_weight
from src.tools.merl_workflow.write_merl_binary import MLP_to_MERL, load_mlp_model
from src.tools.merl_workflow.utils import index_1D_to_3D_tuple, current_datetime
from src.pytorch.model_factory import Trainer
from src.pytorch.dataset.measure_brdf_dataset import MERLDataset
from src.pytorch.dataset.nbrdf_weights_dataset import NBDRFWeightsDataset
from src.tools.merl_workflow.write_merl_binary import write_merl_binary, reverse_transform
from eval.metrics import stack_BRDF_binary_batch

dataset_name = "merl"
# dataset_name = "rgl"
merl_folder_path = 'data/merl/'
merl_binary_name = 'blue-metallic-paint'
output_folder_path = 'output/merl/'
generation_folder_path = 'output/generation/'
file_index = 1
dataset_count = 0


color_scale = [1 / 1500, 1.15 / 1500, 1.66 / 1500]

class PytorchModel(Enum):
    ''' Pytorch model type'''
    MLP = 1
    DIFFUSION = 2

trainer = Trainer()
pytorchModel = PytorchModel.MLP
model = None
dataloader = None
model_weights_path = None

PRINT_TRAIN_BATCH = False
PRINT_PRED_GT = False
print_train_loss_interval = 1
print_eval_loss_interval = 1

# global index array
dataset = None


def setup_trainer():
    global trainer, model, dataset
    trainer = Trainer()
    if pytorchModel == PytorchModel.MLP:
        model = trainer.set_model_MLP()
    elif pytorchModel == PytorchModel.DIFFUSION:
        model = trainer.set_model_hyper_diffusion()
        # initial static NBRDF weights dataset 
        dataset = NBDRFWeightsDataset()


def mean_absolute_logarithmic_error(
        y_true, 
        y_pred):
    # return torch.mean(torch.abs(torch.log(1.1 + y_true) - torch.log(1.1 +
    # y_pred)))
    y_pred_ = y_pred.clone()
    y_true_ = y_true.clone()
    return torch.abs(torch.log(1.1 + y_true_) - torch.log(1.1 + y_pred_))


def neural_network_training_inference(epoch_ini=0):
    '''
        Train and infer the neural network model

        @param  epoch_ini: epoch to start training.

        @return epoch_end: None if default, ending epoch number otherwise.
    '''
    global model, dataloader 
    
    infer_loss_arr = []
    train_loss_arr = []
    # acc_arr = []
    
    batch_size = trainer.get_batch_size()  
    num_epochs = trainer.get_num_epochs()
    optimizer = trainer.get_optimizer()

    if pytorchModel == PytorchModel.DIFFUSION:
        scheduler = trainer.get_scheduler(optimizer)
        stop_loss_threshold = trainer.get_loss_threshold()

    if pytorchModel == PytorchModel.MLP and True:
        # ENABLED: YES     
        model = trainer.load_model("model/mlp_weights_ini.pth")
        print("Using the same initial weights for MLP")
    
    epoch_end = None

    for epoch in range(epoch_ini, num_epochs):        
        model.train()
        loss_batch = []

        dataset.update_type(type=2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch_idx, sample_batch in enumerate(dataloader):
            # Forward pass for each training batch

            if pytorchModel == PytorchModel.MLP:
                X_pos_train_torch, Y_brdf_GT_train_torch = sample_batch

                Y_brdf_predict_torch = model(X_pos_train_torch)
                
                # assert(isinstance(Y_brdf_GT_train_torch, 
                #                   Float[torch.Tensor, str(trainer.get_batch_size()) + " 3"]))
                loss_array = mean_absolute_logarithmic_error(
                    Y_brdf_GT_train_torch, Y_brdf_predict_torch)
                # loss_array = loss_array * torch.Tensor(color_scale).to(device)
                loss = torch.mean(loss_array)
                # loss = loss_function(Y_brdf_predict_torch, Y_brdf_GT_train_torch)

            elif pytorchModel == PytorchModel.DIFFUSION:
                X_mlp_w_train_torch, y_label = sample_batch
                
                # Sample a diffusion timestep
                t = (
                    torch.randint(0, high=model.diff.num_timesteps, size=(X_mlp_w_train_torch.shape[0],))
                    .long()
                    .to(device)
                )
                
                model_kwargs = {
                    'y': y_label,
                    'train': True,
                }

                # compute loss for the sampled timestep
                loss_terms = model.diff.training_losses(
                    model.model,
                    X_mlp_w_train_torch * model.normalization_factor,
                    t,
                    model.mlp_kwargs, 
                    None,
                    model_kwargs=model_kwargs,
                )
       
                loss = loss_terms["loss"].mean()

            if PRINT_PRED_GT and epoch == num_epochs - 1 and pytorchModel == PytorchModel.MLP:
                # Debugging:
                # write the last epoch prediction and ground truth to file
                file = open(output_folder_path + merl_binary_name +
                    "/predict.txt",
                    "w+")
                outputs_1 = Y_brdf_predict_torch.detach().cpu().numpy()
                outputs_2 = Y_brdf_predict_torch.detach().cpu().numpy()
                GT_color_2 = Y_brdf_GT_train_torch.detach().cpu().numpy()
                loss_2 = loss_array.detach().cpu().numpy()
                file.write("predict  |  GT  | loss \n")
                for output_idx, output in enumerate(outputs_2):
                    # print the first 100 records
                    if output_idx > 100:
                        break
                    file.write(  # str(outputs_1[output_idx]) + " " +
                        str(output) + " " +
                        str(GT_color_2[output_idx]) + " " +
                        str(loss_2[output_idx]) + "\n")
                file.close()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())

            if PRINT_TRAIN_BATCH and (
                    batch_idx + 1) % print_train_loss_interval == 0:
                print(
                    f"Batch number [{batch_idx+1}/{batch_size}], train Loss:{loss.item():.4f}")
        

        train_loss_arr.append(np.mean(loss_batch))
        
        if pytorchModel == PytorchModel.DIFFUSION:
            # scheduler.step(loss)
            scheduler.step()

        if (epoch + 1) % print_train_loss_interval == 0:
            print(f"Train epoch:{epoch+1}")
            print(f"Epoch number [{epoch+1}/{num_epochs}], train Loss:{np.mean(loss_batch):.4f}")

            if pytorchModel == PytorchModel.DIFFUSION:
                print("Lr:{:.2E}".format(scheduler.get_last_lr()[0]))

                # early stopping training process if loss is below threshold
                if np.mean(loss_batch) < stop_loss_threshold and True and epoch > 2:
                    print("Stop training with train Loss:{:.4f}".format(np.mean(loss_batch)))
                    epoch_end = epoch + 1
                    break
            
        del dataloader
        gc.collect

        # perform inference step
        model.eval()
        dataset.update_type(type=3)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # no call of Tensor.backward(), disabling computation graph
        with torch.no_grad():

            if pytorchModel == PytorchModel.MLP:

                val_loss = 0.0
                # val_correct = 0
                for batch_idx, sample_batch in enumerate(dataloader):
                    
                    X_pos_infer_torch, Y_brdf_GT_infer_torch = sample_batch

                    # Forward pass
                    Y_brdf_predict_torch = model(X_pos_infer_torch)

                    loss = torch.mean(mean_absolute_logarithmic_error
                            (Y_brdf_GT_infer_torch, Y_brdf_predict_torch))
                    val_loss = loss.item()      # * X_pos_infer_torch.size(0)
                    
                # print validation metrics
                if (epoch + 1) % print_eval_loss_interval == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], val loss:{val_loss:.4f}')

                infer_loss_arr.append(val_loss)

            elif pytorchModel == PytorchModel.DIFFUSION:
                # TODO: Add quick/proper eval here 
                pass

    return train_loss_arr, infer_loss_arr, model, epoch_end # acc_arr


def train_infer_mlp(lr, binary_name, file_index=1):
    '''
        train and infer one MLP model for the given tabulated MERL BRDF
    '''
    global trainer, dataset 

    merl_binary_name = binary_name
    # preprocess_nn_data 
    dataset = MERLDataset(  merl_folder_path, 
                            merl_binary_name, 
                            file_index=file_index)
    trainer.ini_optimizer(lr=lr)
    train_loss_arr, infer_loss_arr, model, _ = neural_network_training_inference()

    # save train_loss_arr to csv
    np.savetxt(
        output_folder_path + merl_binary_name + "/loss"  + ".csv",
        train_loss_arr, 
        delimiter=",")
    # save infer_loss_arr to csv
    np.savetxt(
        output_folder_path + merl_binary_name + "/infer_loss"  + ".csv",
        infer_loss_arr, 
        delimiter=",")

    trainer.save_model(output_folder_path + merl_binary_name + "/model" + ".pth")
    trainer.load_model(output_folder_path + merl_binary_name + "/model" + ".pth")
    print(
        "Training " + str(dataset_count) + " " + binary_name + " finished")
    
    return 0

def train_infer_mlp_loop(from_list=True, file_index=1):
    ''' 
        train and infer MLP model for all the MERL tabulated data
        @param  file_index >  6: index of the interpolated file
                file_index == 1: use the original data
                file_index <= 6: RGB permuted data
    '''

    global dataset_count

    learning_rate = [ 0.05, 0.005, 0.0005, 0.00005, 0.000005]

    if from_list:
        merl_data_list = read_merl_mixed_list(file_index=file_index)
    else:
        merl_data_list = ["blue-metallic-paint"]

    lr = trainer.get_lr()
    lr = 5e-3               # TODO: verify and remove setting fixed value here

    binary_name = merl_data_list[0]

    res = 0

    # for lr in learning_rate:          
        # ''' loop through learning rate lists '''
    for binary_name in merl_data_list:     
        ''' loop through data lists '''
        dataset_count += 1
        print(str(dataset_count) + " " + binary_name)
        # # t finished index
        # if dataset_count <= 93:
        #     continue
        res += train_infer_mlp(lr, binary_name, file_index)   
    # All Finished!
    return res


def train_diffusion(keep_on=False):
    '''
        train HyperDiffusion model with proper setup and saving
    '''
    global trainer, model 

    # learning rate for HyperDiffusion
    lr = trainer.get_lr()
    trainer.ini_optimizer(lr=lr)
    epoch_ini = 0

    if keep_on:
        model, epoch_ini, loss = trainer.load_checkpoint(generation_folder_path+f"checkpoint_hyper_diffusion.pth")
        # learning rate decay
        lr = trainer.get_lr()
        trainer.ini_optimizer(lr=lr)

    train_loss_arr, infer_loss_arr, model, epoch_end = neural_network_training_inference(epoch_ini=epoch_ini) # 0.005 acc_arr
    
    # All Finished! Save model and checkpoint; Plot loss
    trainer.save_model(generation_folder_path+f"model_hyper_diffusion.pth")

    if train_loss_arr is None or len(train_loss_arr) == 0:
        print("Empty train loss array")
    trainer.save_checkpoint(generation_folder_path+f"checkpoint_hyper_diffusion.pth", train_loss_arr[-1], epoch_end)
    
    if infer_loss_arr is None or len(infer_loss_arr) == 0:
        # save train_loss_arr to csv
        np.savetxt(generation_folder_path + f"loss.csv", train_loss_arr, delimiter=",")
    else:
        # save train_loss_arr to csv
        np.savetxt(generation_folder_path + f"loss.csv", train_loss_arr, delimiter=",")
    return 0


def sample_diffusion(from_checkpoint=False):
    '''
        sample HyperDiffusion model from the saved model.
    '''
    if not from_checkpoint:  
        # from model weights
        # saved_hyper_diff_model_name = "model_hyper_diffusion_cond_1000epoch"
        if model_weights_path == None:
            model = trainer.load_model(generation_folder_path + f"model_hyper_diffusion.pth")
        else:
            # Added
            model = trainer.load_model(model_weights_path)
    else:
        # from a checkpoint with more than model weights
        if model_weights_path == None:
            model, _, _ = trainer.load_checkpoint(generation_folder_path + f"checkpoint_hyper_diffusion.pth")
        else:
            # Added
            model = trainer.load_checkpoint(model_weights_path)

    if pytorchModel == PytorchModel.DIFFUSION:
        # merl data
        y = None
        if True:
            print("Inference dataset: MERL")
            if dataset_name == "merl":
                gen_data_size = 100
            elif dataset_name == "rgl":
                gen_data_size = 51
        # inference data 
        if False:
            print("Dataset: Inference")
            dataset.update_type(type=3)
            gen_data_size = 120

        # e.g., extend y to 5 times larger y0, y0, ..., y0, y1, ... y1, ... yN, ... yN
        if False:
            x_times_larger = 5 # 5, 100
            print("Duplicate conditions")
            labels = None
            for label in y:
                if isinstance(label, torch.Tensor):
                    y_ = torch.vstack([label for _ in range(x_times_larger)])
                    labels = y_ if labels is None else torch.vstack([labels, y_])
                else:
                    y_ = np.vstack([label for _ in range(x_times_larger)])
                    labels = y_ if labels is None else np.vstack([labels, y_])
            print(labels.shape)
            y = labels[0:100]
            # y = labels[700:800]

        #  Print inference stat summary
        if y is None:
            print(f"Inference: Unconditional size: {gen_data_size}")
        else:
            print(f"Inference: Condition label size : {gen_data_size}")

        model_kwargs = {
            'y': y,
            'train': False,
        }


        x_0s = (
            model.diff.ddim_sample_loop(model=model.model, 
                                        shape=(gen_data_size, *model.image_size[1:]),
                                        device=device,
                                        model_kwargs=model_kwargs,
                                        clip_denoised=False,
                                        progress=False,
                                        eta=0
                                        ).cpu().float()
        )
        x_0s = x_0s / model.normalization_factor

    # print(x_0s.shape)
    print(x_0s)
    # count non-zero values
    # print(torch.count_nonzero(x_0s, dim=1))
    
    # write to file 
    # check type

    if not isinstance(x_0s, np.ndarray): 
        x_0s = x_0s.numpy()
    np.save(generation_folder_path + f"mlp_weights_new.npy", x_0s)

 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Plot mosaic figure from rendered results.')
    parser.add_argument('-f', '--file_index', type=int, default=-1, help='file index to plot')
    parser.add_argument('-m', '--pytorch_model_type', type=int, default=2, help='pytorch model type to use')
    parser.add_argument("--model_weights_path", type=str, default="model/NeuMaDiff-diversity.pth", help="path to the model checkpoint")
    parser.add_argument("--from_list", type=int, default=0, help="NeuMERL training: single object or from the whole MERL list")
    parser.add_argument("--sample", type=int, default=0, help="Sample from the diffusion model; otherwise train the model.")

    args = parser.parse_args()
    file_index = args.file_index
    pytorchModel = PytorchModel(args.pytorch_model_type)
    model_weights_path = args.model_weights_path
    from_list = (args.from_list != 0)
    
    if file_index > 0:
        output_folder_path = output_folder_path + "merl_"+str(file_index)+"/"

    setup_trainer()

    if pytorchModel == PytorchModel.MLP:

        train_infer_mlp_loop(from_list=from_list, file_index=file_index)

    elif pytorchModel == PytorchModel.DIFFUSION:
        
        if (args.sample == 0):
            train_diffusion(keep_on=False)
        else:
            sample_diffusion(from_checkpoint=False)

            

