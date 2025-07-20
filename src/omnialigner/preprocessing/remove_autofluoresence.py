import sys
import os
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import OrderedDict
import dask.array as da
import numpy as np
import torch
import torch.nn as nn

import omnialigner as om
from omnialigner.logging import logger as logging

root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
vendor_path = os.path.join(root_path, 'vendor/PICASSO/src/')
sys.path.append(root_path)
sys.path.append(vendor_path)
from omnialigner.vendor.PICASSO.src.picasso.nn_picasso import PICASSOnn

def rmaf(om_data: om.Omni3D,
         i_layer: int,
         model_af: str="picasso",
         dict_layer_models: Dict[int, OrderedDict]=None,
         percentile: float=99,
         mi_weight: float=0.9,
         iter_lsqfits: int=-1,
         delta_corr: float=0.05,
         corr_cutoff: float=0.3,
         sink_channels: List[int]=None,
         zoom_level_training:int = None,
         force_training: bool=False,
         source_channel: int=-1) -> da.Array:
    """
    Remove autofluorescence from the image.

    Args:
        omni:                 Omni3D object.
        i_layer:              Index of the image to remove autofluorescence from.
        model_af:             Model to use for removing autofluorescence.
        dict_layer_models:    Dictionary of layer models.
        percentile:           Percentile to use for scaling the maximum.
        mi_weight:            Mutual information weight to use for training the model.
        iter_lsqfits:         Number of iterations to use for least squares fits.
        sink_channels:        Channels to use for removing autofluorescence.
        zoom_level_training:  Zoom level to use for training the model.
        force_training:       If True will train a new model and save it.
                                If False will use cached rmaf model,
        source_channel:       Source channel to use for removing autofluorescence.
    
    Returns:
        da.Array: Image with autofluorescence removed.
    """
    
    dtype = om_data.proj_info.get_dtype(i_layer=i_layer)
    
    zoom_level = om_data.zoom_level if zoom_level_training is None else zoom_level_training
    da_wsi = om_data.load_tiff(i_layer, zoom_level=zoom_level)
    
    if dict_layer_models is None:
        dict_layer_models = {}

    sample_name = om_data.proj_info.get_sample_name(i_layer=i_layer)
    if dtype == "HE":
        return da_wsi
    
    logging.info(f"Removing autofluorescence for {sample_name}")
    
    dir_out = f"{om_data.proj_info.root_dir}/analysis/{om_data.proj_info.project}/{om_data.proj_info.version}/07.fetch_raw/{om_data.proj_info.group}/rmaf"
    if not os.path.exists(dir_out):
        os.makedirs(dir_out, exist_ok=True)
    
    rmaf_params_file = f"{dir_out}/{sample_name}_zoom{zoom_level}_picassoNN.pt"

    
    if len(dict_layer_models) == 0 and os.path.exists(rmaf_params_file):
        logging.info(f"Model exists! Loading model from {rmaf_params_file}")
        dict_layer_models = torch.load(rmaf_params_file, map_location=torch.device('cpu'))
    else:
        logging.info(f"Model {rmaf_params_file} does not exist! Training model")

    # If training a new model, clear the dictionary of layer models
    if force_training:
        dict_layer_models = {}

    da_unmixed, dict_layer_models = rmaf_dask(
            da_wsi,
            model_af,
            dict_layer_models=dict_layer_models,
            percentile=percentile,
            mi_weight=mi_weight,
            iter_lsqfits=iter_lsqfits,
            delta_corr=delta_corr,
            corr_cutoff=corr_cutoff,
            sink_channels=sink_channels,
            source_channel=source_channel
    )
    
    if not os.path.exists(rmaf_params_file) or force_training:
        torch.save(dict_layer_models, rmaf_params_file, _use_new_zipfile_serialization=False)
    
    return da_unmixed

def calculate_percentile_max(da_wsi: da.Array, percentile: float=99, max_size: int=5000) -> np.ndarray:
    l_percentile_max = []
    for i in range(da_wsi.shape[2]):      
        np_arr = da_wsi[:,:,i].compute().astype(np.float32)
        if np_arr.shape[0] > max_size:
            np_arr = cv2.resize(np_arr, (max_size, max_size))

        percentile_max = np.percentile(np_arr[np_arr>0], percentile)
        l_percentile_max.append(percentile_max)
    return np.array(l_percentile_max)

def scale_maximum(da_wsi: da.Array, percentile: float=99, l_percentile_max: List=None) -> da.Array:
    l_stacks = []
    if l_percentile_max is None:
        l_percentile_max = calculate_percentile_max(da_wsi, percentile)

    for i in range(da_wsi.shape[2]):      
        np_arr = da_wsi[:,:,i].compute().astype(np.float32)
        percentile_max = l_percentile_max[i]
        np_new = np.clip(255*np_arr / percentile_max, 0, 255).astype(np.uint8)
        if np.mean(np_new-np.mean(np_arr)) >= 25.5:
            np_new = np_arr

        l_stacks.append(np_new)

    return np.stack(l_stacks, axis=2)


def rmaf_dask(da_wsi: da.Array,
              model_af: str="picasso",
              dict_layer_models: Dict[int, OrderedDict]=None,
              percentile: float=99,
              mi_weight: float=0.99,
              iter_lsqfits: int=-1,
              delta_corr: float=0.05,
              corr_cutoff: float=0.3,
              sink_channels: List[int]=None,
              source_channel: int=-1) -> Tuple[da.Array, Dict[int, OrderedDict]]:
    if sink_channels is None:
        sink_channels = list(range(1, da_wsi.shape[2]-1))


    if model_af == "picasso":
        np_unmixed, out_dict_models = rmaf_dask_picasso(
                da_wsi,
                dict_layer_models=dict_layer_models,
                percentile=percentile,
                mi_weight=mi_weight,
                iter_lsqfits=iter_lsqfits,
                delta_corr=delta_corr,
                corr_cutoff=corr_cutoff,
                sink_channels=sink_channels,
                source_channel=source_channel
        )
    else:
        raise ValueError(f"Model {model_af} not supported")
    return da.from_array(np_unmixed), out_dict_models


def rmaf_dask_picasso(da_wsi: da.Array,
            dict_layer_models: Dict[int, nn.Module]=None,
            percentile: float=99,
            mi_weight=0.9,
            iter_lsqfits: int=-1,
            delta_corr: float=0.05,
            corr_cutoff: float=0.3,
            sink_channels: List[int]=None,
            source_channel: int=-1) -> Tuple[da.Array, np.ndarray, Dict[int, nn.Module]]:

    if dict_layer_models is None:
        dict_layer_models = {}

    if "percentile_max" not in dict_layer_models:
        l_percentile_max = calculate_percentile_max(da_wsi, percentile)
        dict_layer_models["percentile_max"] = l_percentile_max
    
    np_scaled = scale_maximum(da_wsi, percentile, dict_layer_models["percentile_max"])
    np_unmixed = np_scaled.copy()
    
    dict_models_new = {"percentile_max": dict_layer_models["percentile_max"]}
    source_im = np_scaled[:, :, source_channel]
    for i_sink in tqdm(sink_channels):
        mixing_matrix = np.array([[1],[-1]])
        sink_im = np_scaled[:, :, i_sink]

        if i_sink not in dict_layer_models:
            logging.info(f"Training model for sink channel {i_sink}")
            model = train_rmaf(source_im, sink_im, mixing_matrix, mi_weight=mi_weight)
        else:
            model = dict_layer_models[i_sink]

        dict_models_new[i_sink] = model
        mixing_parameters = model.mixing_parameters
        alpha = mixing_parameters[0,:]
        bg = mixing_parameters[1,:].T

        ims_ = np.array([sink_im.flatten(), source_im.flatten()]).T
        unmixed_im = (ims_ - bg).clip(min=0).dot(alpha)
        unmixed_im = unmixed_im.clip(min=0).astype('int16')
        unmixed_sink = unmixed_im[:,0].reshape((sink_im.shape[0],-1)).squeeze()
        if iter_lsqfits > 0:
            unmixed_sink = iterative_remove_co_linear(
                source_im.flatten(), unmixed_sink.flatten(),
                max_iter=iter_lsqfits
            )[0].reshape(sink_im.shape[0], -1).squeeze()

        np_unmixed[:, :, i_sink] = unmixed_sink

    return np_unmixed, dict_models_new

def train_rmaf(source_im, sink_im, mixing_matrix, mi_weight=0.9):
    print("Training RMAF model..., mi_weight:", mi_weight)
    model = PICASSOnn(mixing_matrix, mi_weight=mi_weight)
    for _ in model.train_loop([sink_im, source_im]):
        pass
    
    return model


def remove_co_linear_torch(x, y):
    """
    lsq to remove y with respect to x.

    Args:
        x: Tensor, shape [N]
        y: Tensor, shape [N]

    Returns:
        y_fit: Co-linear fit vector
        y_residual: resdule vector
        coeffs: Tensor, shape [2], 拟合系数 (a, b)
    """
    x = x.flatten().float()
    y = y.flatten().float()

    N = x.shape[0]
    A = torch.stack([x, torch.ones_like(x)], dim=1)  # shape: [N, 2]

    
    AtA = A.T @ A  # shape: [2, 2]
    Aty = A.T @ y  # shape: [2]
    coeffs = torch.linalg.solve(AtA, Aty)  # shape: [2]

    y_fit = (A @ coeffs).clamp(0, 255)
    y_residual = (y - y_fit).clamp(0, 255)

    return y_residual, y_fit, coeffs


def iterative_remove_co_linear(x: torch.FloatTensor|np.ndarray, y: torch.FloatTensor|np.ndarray, max_iter:int=10, delta_corr:float=0.02, corr_cutoff:float=0.3) -> Tuple[torch.FloatTensor|np.ndarray, torch.FloatTensor|np.ndarray, List[torch.FloatTensor]]:
    """
    Iteratively remove the co-linear part of y with respect to x using least squares residuals.

    Args:
        x: Tensor, shape [N]
        y: Tensor, shape [N]
        max_iter: Maximum number of iterations
        delta_corr: Early stopping if the sum of fitted values in an iteration is below this threshold

    Returns:
        final_residual: The final residual (y after decontamination)
        total_fit: The total fitted component removed from y
        coeffs_list: List of (a, b) coefficients for each iteration
    """
    ret_np = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
        ret_np = True

    x = x.flatten().float()
    y_res = y.flatten().float().clone()
    total_fit = torch.zeros_like(y_res)
    coeffs_list = []

    pre_corr = 1.0
    for i in range(max_iter):
        y_res_iter, y_fit, coeffs = remove_co_linear_torch(x, y_res)
        total_fit = y_fit
        coeffs_list.append(coeffs.detach().cpu())

        pearson_corr = np.corrcoef(x.detach().cpu().numpy(), y_res_iter.detach().cpu().numpy())[0, 1]
        if np.isnan(pearson_corr):
            break

        print(f"Iteration {i+1}, Pearson correlation: {pearson_corr:.4f}, Coefficients: {coeffs.detach().cpu().numpy()}")
        delta = pre_corr-pearson_corr
        if delta < delta_corr or pearson_corr < corr_cutoff:
            break

        pre_corr = pearson_corr
        y_res = y_res_iter

    if ret_np:
        y_res = y_res.detach().cpu().numpy()
        total_fit = total_fit.detach().cpu().numpy()

    return y_res, total_fit, coeffs_list


def apply_coeffs_list(x: np.ndarray|torch.FloatTensor, y0: np.ndarray|torch.FloatTensor, coeffs_list: List[torch.FloatTensor]) -> torch.FloatTensor|np.ndarray:
    """
    Apply a list of coefficients to the input x, returning the total fit and per-iteration fits.
    
    Args:
        x: Tensor, shape [N]
        y0: Tensor, shape [N], raw input data
        coeffs_list: List[Tensor of shape [2]], coefficients of each iter (a, b) calculated by iterative_remove_co_linear
        
    Returns:
        total_fit: Tensor, shape [N], 
        fit_per_iter: List[Tensor], fit per iteration, each of shape [N]
    """
    ret_np = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y0, np.ndarray):
        y0 = torch.from_numpy(y0).float()
        ret_np = True
    
    x = x.flatten().float()
    y = y0.flatten().float()
    A = torch.stack([x, torch.ones_like(x)], dim=1)  # [N, 2]

    for coeffs in coeffs_list:
        coeffs = coeffs.to(x.device).float()
        y_fit = (A @ coeffs).clamp(0, 255)
        y = (y - y_fit).clamp(0, 255)

    if ret_np:
        y = y.detach().cpu().numpy()

    return y
