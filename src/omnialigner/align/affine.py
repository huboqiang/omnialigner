import os
import sys
from typing import Tuple, List
from collections import OrderedDict
import torch
import numpy as np

import omnialigner as om
from omnialigner.align.models.aligner import OmniAligner, train_model
from omnialigner.align.nonrigid import get_keypoints_pairs
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageTag
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw

def affine(
        om_data: om.Omni3D, 
        l_layers: List[int]=None, 
        overwrite_cache:bool=False
    ) -> Tuple[Tensor_image_NCHW, List[Tensor_kpts_N_xy_raw]]:
    """Perform affine alignment on a sequence of images.

    This function applies affine transformation to align a sequence of images using keypoints
    and optimizes the transformation parameters through a neural network model. The alignment
    process consists of several stages:

    1. Input validation: Checks for required padded and stacked tensors
    2. Cache handling: Returns cached results if available and not overwriting
    3. Layer processing: Expands layer indices if not specified
    4. Model training: Uses OmniAligner to optimize transformation parameters
    5. Cache saving: Stores results for future use

    Args:
        om_data (om.Omni3D): Omni3D object containing:
            - Image data and keypoints
            - Project configuration
            - Maximum image size
            - Other alignment parameters
        l_layers (List[int], optional): List of layer indices to process. 
            If None, processes all layers. Defaults to None.
        overwrite_cache (bool, optional): Whether to overwrite existing cached results.
            If True, forces recomputation even if cached results exist.
            Defaults to False.

    Returns:
        tuple:
            - aligned_tensor (Tensor_image_NCHW): Aligned image tensor with shape:
                [N, C, H, W] where:
                - N: number of layers
                - C: number of channels
                - H, W: height and width after padding
            - l_kpts_moved (List[Tensor_kpts_N_xy_raw]): List of transformed keypoints,
                where each element is a tensor of shape [K, 2] containing K keypoint
                coordinates (x, y) for each layer.

    Raises:
        RuntimeError: If required padded or stacked tensors are not found in cache.

    Note:
        - This function requires prior execution of `om.pad()` and `om.stack()`.
        - The alignment process uses a neural network model (OmniAligner) to optimize
          the affine transformation parameters.
        - Results are cached by default to avoid redundant computation.
        - The function handles both regular stacking and transformation-based stacking.
        - Initial transformation parameters are set based on padding ratios.
        - The model training process can be customized through the configuration
          dictionary in om_data.config["align"].

    Example:
        >>> import omnialigner as om
        >>> om_data = om.Omni3D(project_dir="path/to/project")
        >>> om.pad(om_data)  # Required preprocessing
        >>> om.stack(om_data)  # Required preprocessing
        >>> aligned_tensor, moved_keypoints = om.affine(om_data)
    """
    
    logging.info("Processing affine, stage1, check inputs")
    dict_file_name_pad = StageTag.PAD.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_pad is None:
        logging.error("Padded tensor not found. Please run `om.pad()` first.")
        return None, None

    
    dict_file_name_stack = StageTag.STACK.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_stack is None:
        logging.error("Stacked tensor not found. Please run `om.stack()` first.")
        return None, None
    
    dir_stack = os.path.dirname(dict_file_name_stack["padded_tensor"])
    is_stacked_using_tfrs = False
    if os.path.isfile(f"{dir_stack}/omni_stack/tfrs_to_stacked.pt"):
        is_stacked_using_tfrs = True


    l_ratios = torch.load(dict_file_name_pad["l_ratio"])
    l_init_trs = []
    for i in range(len(om_data)):
        ratio = l_ratios[i] if not is_stacked_using_tfrs else 1.0
        trs = torch.FloatTensor([0, 0, 0, np.log(ratio), np.log(ratio)])
        l_init_trs.append(trs)

    l_init_trs = torch.stack(l_init_trs)
    logging.info("Processing affine, stage2, return cache if exists")
    dict_file_name_affine = StageTag.AFFINE.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_affine is not None and not overwrite_cache:
        l_kpts_moved = torch.load(dict_file_name_affine["affine_kpts"])
        aligned_tensor = torch.load(dict_file_name_affine["padded_tensor"])
        if l_layers is not None:
            l_kpts_moved = [l_kpts_moved[i_layer] for i_layer in l_layers]
            aligned_tensor = aligned_tensor[l_layers, :, :, :]
        return aligned_tensor, l_kpts_moved

    logging.info("Processing affine, stage3, expand l_layers")
    if l_layers is None:
        N_layers = len(om_data)
        l_layers = range(N_layers)
    else:
        overwrite_cache = False

    dict_file_name_affine = StageTag.AFFINE.get_file_name(projInfo=om_data.proj_info, check_exist=False)
    logging.info("Processing affine, stage4, run")
    padded_tensors = torch.load(dict_file_name_stack["padded_tensor"])[l_layers, :, :, :]
    l_kpt_pairs = torch.load(dict_file_name_stack["l_kpts_pairs"])
    out_dir_affine = dict_file_name_affine["out_dir"]
    os.makedirs(out_dir_affine, exist_ok=True)

    ## Just one level of rank
    l_kpt_pairs_ = [ [[l_kpt_pairs[idx][0]/om_data.max_size, l_kpt_pairs[idx][1]/om_data.max_size]] for idx in l_layers[:-1] ]

    image_3d_tensor_mean = torch.mean(padded_tensors, (1))[:, None, :, :]
    dict_config = om_data.config["align"]
    model = OmniAligner(
        image_3d_tensor=image_3d_tensor_mean, 
        l_kpt_pairs=l_kpt_pairs_, 
        log_prefix=out_dir_affine, 
        l_init_trs=l_init_trs,
        dict_config=dict_config,
        model_type="affine",
        save_prefix="affine"
    )

    n_epochs = dict_config["affine"].get("used_levels", 1)
    train_model(model, num_epochs=n_epochs)

    logging.info("Align affine done")
    dev = torch.device("cpu")
    model.image_3d_tensor = padded_tensors
    model.tensor_size = [padded_tensors.shape[2], padded_tensors.shape[3]]
    for mod in model.grid2d_modules:
        mod.set_device(dev)
        mod.tensor_size = [padded_tensors.shape[2], padded_tensors.shape[3]]

    model.eval()
    model.dev = dev
    model.n_keypoints = -1

    logging.info("generating tensor")

    aligned_tensor, l_kpts_moved = model.viz_all(padded_tensors, l_kpt_pairs=l_kpt_pairs_)
    dict_model = OrderedDict()
    for idx in range(model.N):
        dict_model[f"{idx}.tensor_trs"] = model.grid2d_modules[idx].tensor_trs

    logging.info("generating tensor done")
    logging.info(f"Processing affine, stage5, save cache {overwrite_cache}")
    if overwrite_cache or not os.path.exists(dict_file_name_affine["affine_model"]):
        torch.save(aligned_tensor, dict_file_name_affine["padded_tensor"], _use_new_zipfile_serialization=False)
        torch.save(dict_model, dict_file_name_affine["affine_model"], _use_new_zipfile_serialization=False)
        torch.save(l_kpts_moved, dict_file_name_affine["affine_kpts"], _use_new_zipfile_serialization=False)

    return aligned_tensor, l_kpts_moved




def affine(
        om_data: om.Omni3D, 
        l_layers: List[int]=None, 
        overwrite_cache:bool=False
    ) -> Tuple[Tensor_image_NCHW, List[Tensor_kpts_N_xy_raw]]:
    """Perform affine alignment on a sequence of images.

    This function applies affine transformation to align a sequence of images using keypoints
    and optimizes the transformation parameters through a neural network model. The alignment
    process consists of several stages:

    1. Input validation: Checks for required padded and stacked tensors
    2. Cache handling: Returns cached results if available and not overwriting
    3. Layer processing: Expands layer indices if not specified
    4. Model training: Uses OmniAligner to optimize transformation parameters
    5. Cache saving: Stores results for future use

    Args:
        om_data (om.Omni3D): Omni3D object containing:
            - Image data and keypoints
            - Project configuration
            - Maximum image size
            - Other alignment parameters
        l_layers (List[int], optional): List of layer indices to process. 
            If None, processes all layers. Defaults to None.
        overwrite_cache (bool, optional): Whether to overwrite existing cached results.
            If True, forces recomputation even if cached results exist.
            Defaults to False.

    Returns:
        tuple:
            - aligned_tensor (Tensor_image_NCHW): Aligned image tensor with shape:
                [N, C, H, W] where:
                - N: number of layers
                - C: number of channels
                - H, W: height and width after padding
            - l_kpts_moved (List[Tensor_kpts_N_xy_raw]): List of transformed keypoints,
                where each element is a tensor of shape [K, 2] containing K keypoint
                coordinates (x, y) for each layer.

    Raises:
        RuntimeError: If required padded or stacked tensors are not found in cache.

    Note:
        - This function requires prior execution of `om.pad()` and `om.stack()`.
        - The alignment process uses a neural network model (OmniAligner) to optimize
          the affine transformation parameters.
        - Results are cached by default to avoid redundant computation.
        - The function handles both regular stacking and transformation-based stacking.
        - Initial transformation parameters are set based on padding ratios.
        - The model training process can be customized through the configuration
          dictionary in om_data.config["align"].

    Example:
        >>> import omnialigner as om
        >>> om_data = om.Omni3D(project_dir="path/to/project")
        >>> om.pad(om_data)  # Required preprocessing
        >>> om.stack(om_data)  # Required preprocessing
        >>> aligned_tensor, moved_keypoints = om.affine(om_data)
    """
    
    logging.info("Processing affine, stage1, check inputs")
    dict_file_name_pad = StageTag.PAD.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_pad is None:
        logging.error("Padded tensor not found. Please run `om.pad()` first.")
        return None, None

    
    dict_file_name_stack = StageTag.STACK.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_stack is None:
        logging.error("Stacked tensor not found. Please run `om.stack()` first.")
        return None, None
    
    dir_stack = os.path.dirname(dict_file_name_stack["padded_tensor"])
    is_stacked_using_tfrs = False
    if os.path.isfile(f"{dir_stack}/omni_stack/tfrs_to_stacked.pt"):
        is_stacked_using_tfrs = True


    l_ratios = torch.load(dict_file_name_pad["l_ratio"])
    l_init_trs = []
    for i in range(len(om_data)):
        ratio = l_ratios[i] if not is_stacked_using_tfrs else 1.0
        trs = torch.FloatTensor([0, 0, 0, np.log(ratio), np.log(ratio)])
        l_init_trs.append(trs)

    l_init_trs = torch.stack(l_init_trs)
    logging.info("Processing affine, stage2, return cache if exists")
    dict_file_name_affine = StageTag.AFFINE.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_affine is not None and not overwrite_cache:
        l_kpts_moved = torch.load(dict_file_name_affine["affine_kpts"])
        aligned_tensor = torch.load(dict_file_name_affine["padded_tensor"])
        if l_layers is not None:
            l_kpts_moved = [l_kpts_moved[i_layer] for i_layer in l_layers]
            aligned_tensor = aligned_tensor[l_layers, :, :, :]
        return aligned_tensor, l_kpts_moved

    logging.info("Processing affine, stage3, expand l_layers")
    if l_layers is None:
        N_layers = len(om_data)
        l_layers = range(N_layers)
    else:
        overwrite_cache = False

    dict_file_name_affine = StageTag.AFFINE.get_file_name(projInfo=om_data.proj_info, check_exist=False)
    logging.info("Processing affine, stage4, run")
    padded_tensors = torch.load(dict_file_name_stack["padded_tensor"])[l_layers, :, :, :]
    l_kpt_pairs = torch.load(dict_file_name_stack["l_kpts_pairs"])
    out_dir_affine = dict_file_name_affine["out_dir"]
    os.makedirs(out_dir_affine, exist_ok=True)

    ## Just one level of rank
    l_kpt_pairs_ = [ [[l_kpt_pairs[idx][0]/om_data.max_size, l_kpt_pairs[idx][1]/om_data.max_size]] for idx in l_layers[:-1] ]

    image_3d_tensor_mean = torch.mean(padded_tensors, (1))[:, None, :, :]
    dict_config = om_data.config["align"]
    model = OmniAligner(
        image_3d_tensor=image_3d_tensor_mean, 
        l_kpt_pairs=l_kpt_pairs_, 
        log_prefix=out_dir_affine, 
        l_init_trs=l_init_trs,
        dict_config=dict_config,
        model_type="affine",
        save_prefix="affine"
    )

    n_epochs = dict_config["affine"].get("used_levels", 1)
    train_model(model, num_epochs=n_epochs)

    logging.info("Align affine done")
    dev = torch.device("cpu")
    model.image_3d_tensor = padded_tensors
    model.tensor_size = [padded_tensors.shape[2], padded_tensors.shape[3]]
    for mod in model.grid2d_modules:
        mod.set_device(dev)
        mod.tensor_size = [padded_tensors.shape[2], padded_tensors.shape[3]]

    model.eval()
    model.dev = dev
    model.n_keypoints = -1

    logging.info("generating tensor")

    aligned_tensor, l_kpts_moved = model.viz_all(padded_tensors, l_kpt_pairs=l_kpt_pairs_)
    dict_model = OrderedDict()
    for idx in range(model.N):
        dict_model[f"{idx}.tensor_trs"] = model.grid2d_modules[idx].tensor_trs

    logging.info("generating tensor done")
    logging.info(f"Processing affine, stage5, save cache {overwrite_cache}")
    if overwrite_cache or not os.path.exists(dict_file_name_affine["affine_model"]):
        torch.save(aligned_tensor, dict_file_name_affine["padded_tensor"], _use_new_zipfile_serialization=False)
        torch.save(dict_model, dict_file_name_affine["affine_model"], _use_new_zipfile_serialization=False)
        torch.save(l_kpts_moved, dict_file_name_affine["affine_kpts"], _use_new_zipfile_serialization=False)

    return aligned_tensor, l_kpts_moved


if __name__ == '__main__':
    project = sys.argv[1]
    group = sys.argv[2]

    root_dir = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen"
    out_dir = f"{root_dir}/analysis/{project}/05.align3d/{group}/"

    padded_tensors = torch.load(f"{root_dir}/analysis/{project}/04.detect_kpts/{group}/merged_tensor.flip_angles.pt")
    l_kpt_pairs = torch.load(f"{root_dir}/analysis/{project}/04.detect_kpts/{group}/l_kpts_pairs.flip_angles.pt")
