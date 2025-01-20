""" Evaluation metrics for generated BRDFs
    @section Source
    https://github.com/luost26/diffusion-point-cloud/blob/main/evaluation/evaluation_metrics.py

    From https://github.com/stevenygd/PointFlow/tree/master/metrics

    @file
"""

# --- built in ---
import sys
from pathlib import Path

# setting the repo directory to sys path
sys.path.append(str(Path(__file__).parent.parent.parent))

# --- 3rd party ---
import torch
import numpy as np
from tqdm.auto import tqdm
import cv2
import argparse

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import variation_of_information as vi
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- my module ---
from src.tools.merl_workflow.read_merl_data import read_merl_data_flatten, read_merl_interpolated_flatten
from src.tools.merl_workflow.read_merl_list import read_merl_mixed_list
from src.pytorch.utils.device import device
from src.tools.merl_workflow.utils import index_1D_to_3D_tuple, RGB_permute_BRDF
from src.tools.merl_workflow.write_merl_binary import write_merl_binary, reverse_transform

is_brdf_space = True 
# control which metric to use


def mean_absolute_logarithmic_error(y_true, y_pred, type):
    '''
        @param type: 1: log-cos-L1, 2: log-L1, 3: L1

        We found that the result from mlp may exceed reasonable value and cause the later calculation to be inf or nan, hence we cap the value to 1e8
    '''
    y_true[torch.isinf(y_true)] = 1e8
    y_true[torch.isnan(y_true)] = 1e8
    y_pred[torch.isinf(y_pred)] = 1e8
    y_pred[torch.isnan(y_pred)] = 1e8
    
    y_pred_ = y_pred.clone()
    y_true_ = y_true.clone()

    # (1, 3, 1458000) clip all negative values to 0

    y_pred_[y_pred_ < 0] = 0.0
    y_true_[y_true_ < 0] = 0.0
    
    if type == 1:
        add_cos_theta_i= True
    else:
        add_cos_theta_i = False

    if add_cos_theta_i: # cos theta_i
    # generate range from 0 to 1458000
        x = np.arange(0, 1458000)
        X_pos = index_1D_to_3D_tuple(x)
            # Used for 6-dim input
        theta_h = np.square(X_pos[0] / 90.0) * (np.pi/2)
        theta_d = X_pos[1] / 90.0 * (np.pi/2)
        phi_d = X_pos[2] / 180.0 * np.pi

        theta_d = torch.tensor(theta_d).float().to(device)
        theta_h = torch.tensor(theta_h).float().to(device)
        phi_d = torch.tensor(phi_d).float().to(device)
        
        wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
            torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
        y_pred_ = y_pred_ * torch.clamp(wiz, 0, 1)
        y_true_ = y_true_ * torch.clamp(wiz, 0, 1)
    
    if type <= 2:
        brdf_space_distance = torch.abs(torch.log(1.1 + y_true_) - torch.log(1.1 + y_pred_))
    else:
        brdf_space_distance = torch.abs(y_true_ -  y_pred_)
    
    brdf_space_distance[torch.isnan(brdf_space_distance)] = 1e8
    brdf_space_distance[torch.isinf(brdf_space_distance)] = 1e8
    brdf_space_distance[brdf_space_distance > 1e8] = 1e8

    return brdf_space_distance


def graphics_metrics(src, test, type):
    '''
        @param type: 1: SSIM, 2: PSNR, 3: NRMSE, 4: MSE
    '''
    test = test.reshape(256, 256, 3)
    src = src.reshape(256, 256, 3)
    # to numpy and int
    test = test.cpu().detach().numpy().astype(np.uint8)
    src = src.cpu().detach().numpy().astype(np.uint8)
    mse_score = mse(src, test)
    nrmse_score = nrmse(src, test)
    # tombstone for now, self distance is removed at the end
    if mse_score == 0:  
        psnr_score = 0
        ssim_score = 0
    else:
        psnr_score = psnr(src, test, data_range=255)
        ssim_score = ssim(src, test, win_size=3, multichannel=True)
        # negate the score for psnr and ssim
        psnr_score = -psnr_score
        ssim_score = -ssim_score
        
    if type == 1:
        return ssim_score
    elif type == 2:
        return psnr_score
    elif type == 3:
        return nrmse_score
    elif type == 4:
        return mse_score

    
def pairwise_distance(brdf_set_1, brdf_set_2, batch_size, type, is_pairwise=False, verbose=True):
    global is_brdf_space
    N_1 = len(brdf_set_1)       # .shape[0]
    N_2 = len(brdf_set_2)       # .shape[0]
    all_cd = []
    iterator = range(N_1)
    if verbose:
        desc_str = 'Distance'
        if is_pairwise:
            desc_str = 'Pairwise Distance'
        iterator = tqdm(iterator, desc=desc_str)

    for set1_start in iterator:

        set1_batch = brdf_set_1[set1_start]
        cd_lst = []
        sub_iterator = range(0, N_2, batch_size)

        for set2_start in sub_iterator:

            set1_single = brdf_set_1[set1_start]

            set2_batch = brdf_set_2[set2_start: min(N_2, set2_start + batch_size)]
            if is_brdf_space:
                loss = mean_absolute_logarithmic_error(set1_single, set2_batch, type)
                loss = torch.mean(torch.mean(loss, dim=2), dim=1).view(1, -1)
            else:
                loss = graphics_metrics(set1_single, set2_batch, type)

            cd_lst.append(loss)

        if is_brdf_space:
            cd_lst = torch.cat(cd_lst, dim=1)
        
        all_cd.append(cd_lst)

    if is_brdf_space:
        all_cd = torch.cat(all_cd, dim=0) # N_1 * N_2
    else: 
        # convert list to tensor
        all_cd = torch.tensor(all_cd)

    return all_cd 


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    '''
        r: reference
        s: sample
        M: distance matrix
            Mrr     Mrs

            Msr     Mss

        label = {1 if in reference set, 0 if in sample set}
        pred = {count == 1, index by the nearest neighbour index}

        We observe that the reasonable range of L1 loss between two material BRDFs should be less than 1e-6, however 1-NNA count these as valid neighbors. Hence we introduce modified L1.
    '''
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    # 1 ...... 1 (* #ref) 0 ... 0 (* #sample)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat([
        torch.cat((Mxx, Mxy), 1),
        torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)

    if sqrt:
        M = M.abs().sqrt()

    INFINITY = float('inf')
    # eliminate self distance and get top K=1
    M_ = M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))
    val, idx = (M_).topk(
        k, 0, False)

    idx[val >= 1e6] = n1 + 1    # modified 1-NNA: 
    # move those samples with too distant neighbour away from matching to reference set.

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        # count reference, indicator function
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t (50%)': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f (50%)': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc (50%)': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    '''
       @param  all_dist: N_2 * N_1 
        min_val: N_1
        min_val_fromsmp: N_2
    '''
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    print("Matched reference BRDF id indexed by sample BRDF:")
    if min_idx.size(0) > 10:
        print(min_idx.view(10, -1))
    else:
        print(min_idx)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'minimum matching distance (from ref) (↓)': mmd,
        'coverage (↑)': cov,
        'minimum matching distance (from sample)': mmd_smp,
        # 'min_idx': min_idx
    } # , min_idx.view(-1)


def compute_all_metrics(sample_brdf_set, ref_brdf_set, batch_size, type):
    '''
        @param type: 1: SSIM, 2: PSNR, 3: NRMSE, 4: MSE

        Require batch_size = 1 for img
    '''
    results = {}

    M_rs_cd = pairwise_distance(ref_brdf_set, sample_brdf_set, batch_size, type=type, is_pairwise=True) 
    # M_rs_emd 

    ## D
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s" % k: v for k, v in res_cd.items()
    })
    
    ## EMD
    # res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    M_rr_cd = pairwise_distance(ref_brdf_set, ref_brdf_set, batch_size, type) # M_rr_emd
    M_ss_cd = pairwise_distance(sample_brdf_set, sample_brdf_set, batch_size, type) # M_ss_emd 

    # 1-NN results
    ## D
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-Nearest-Neighbour-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    
    ## EMD
    # one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    for k, v in results.items():
        print('[%s]: %.8f' % (k, v.item()))

    return results


def stack_BRDF_binary_batch(merl_data_1D, brdf_set):
    '''
        Stack each single brdf into 2D numpy array
    '''
    if brdf_set is not None:
        brdf_set = np.row_stack((brdf_set, merl_data_1D))
    else:
        brdf_set = merl_data_1D
    return brdf_set


def stack_img_batch(cv2_img, img_set):
    '''
        Stack each cv2 img of shape (1, 256, 256, 3) into array
    '''
    if img_set is not None:
        img_set = np.vstack((img_set, cv2_img))
    else:
        img_set = cv2_img
    return img_set


if __name__ == '__main__':
    verbose = True
    reference_brdf_set = None
    sample_brdf_set = None
    refer_set_size = 1  
    sample_set_size = 1 
    
    brdf_space_metrics = ["Log-cos-L1", "Log-L1", "L1"]
    image_space_metrics = ["SSIM", "PSNR", "NRMSE", "MSE"]

    parser = argparse.ArgumentParser(description='Evaluation metrics for synthetic BRDFs')
    parser.add_argument('--is_brdf_space', 
                        type=int, 
                        default=0, 
                        help='use BRDF space distance as underlying metric, otherwise use image space distance')
    parser.add_argument('--reference_folder_path', 
                        type=str, 
                        default="data/merl/", 
                        help='the path to reference folder')
    parser.add_argument('--sample_folder_path', 
                        type=str, 
                        default="data/merl/", 
                        help='the path to sample folder')
    parser.add_argument('--reference_img_path', 
                        type=str, 
                        default="output/img/",
                        help='the path to reference image folder')
    parser.add_argument('--sample_img_path', 
                        type=str, 
                        default="output/img/", 
                        help='the path to sample image folder')
    parser.add_argument('--refer_set_size', 
                        type=int, 
                        default=1, 
                        help='refer set size')
    parser.add_argument('--sample_set_size', 
                        type=int, 
                        default=1, 
                        help='sample set size')

    
    args = parser.parse_args()
    is_brdf_space = (args.is_brdf_space != 0)
    reference_folder_path = args.reference_folder_path
    sample_folder_path = args.reference_folder_path
    reference_img_path = args.reference_img_path
    sample_img_path = args.sample_img_path
    refer_set_size = args.refer_set_size
    sample_set_size = args.sample_set_size 

    if is_brdf_space:

        ''' Load reference BRDFs'''

        # TODO: please update it to the your own list
        merl_data_list = ["blue-metallic-paint"] 

        iterator = range(refer_set_size) 
        if verbose:
            iterator = tqdm(iterator, desc='Loading reference BRDF')

        for i in iterator:
            file_path = reference_folder_path + merl_data_list[i] + '.binary'
            merl_data_1D = read_merl_data_flatten(file_path).T.reshape(1, 3, 1458000)
            reference_brdf_set = stack_BRDF_binary_batch(merl_data_1D, reference_brdf_set)
        
        ''' Load sample BRDFs'''

        # TODO: please update it to the your own list
        merl_data_list = ["blue-metallic-paint"]

        iterator = range(sample_set_size)
        if verbose:
            iterator = tqdm(iterator, desc='Loading sample BRDF')

        for index in iterator: 
            file_path = sample_folder_path + merl_data_list[i] + '.binary'
            merl_data_1D_gen = read_merl_data_flatten(file_path).T.reshape(1, 3, 1458000)
            sample_brdf_set = stack_BRDF_binary_batch(merl_data_1D_gen, sample_brdf_set)

        metrics = brdf_space_metrics
        max_type = 4

    if not is_brdf_space:

        '''Load reference rendered images'''

        iterator = range(refer_set_size) 
        if verbose:
            iterator = tqdm(iterator, desc='Loading reference BRDF images')
        
        for i in iterator:
            src = cv2.imread(reference_img_path + f"GT_{i}.png")  
            src = src.reshape(1, 256, 256, 3)
            reference_brdf_set = stack_img_batch(src, reference_brdf_set)
 
        print(reference_brdf_set.shape)
       
        '''Load sample rendered images'''

        iterator = range(sample_set_size) 
        if verbose:
            iterator = tqdm(iterator, desc='Loading sample BRDF images')

        for i in iterator:
            src = cv2.imread(f"{sample_img_path}/gen_{i}.png")
            src = src.reshape(1, 256, 256, 3)
            sample_brdf_set = stack_img_batch(src, sample_brdf_set)

        print(sample_brdf_set.shape)

        metrics = image_space_metrics
        max_type = 5

    reference_brdf_set = torch.tensor(reference_brdf_set).float().to(device)
    sample_brdf_set = torch.tensor(sample_brdf_set).float().to(device)

    for i in range(1, max_type):
        print("Type: " + str(i))
        print("Metrics: " + metrics[i-1])
        compute_all_metrics(sample_brdf_set, reference_brdf_set, 1, i)
