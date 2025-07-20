import os
import sys
from typing import Tuple, List
from collections import OrderedDict
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import dask.array as da

import omnialigner as om
from omnialigner.align.models.aligner import OmniAligner, train_model
from omnialigner.preprocessing.pad import get_pad_size, get_ratio
from omnialigner.cache_files import StageTag
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy, Tensor_kpt_pair, Tensor_Layer_kpt_pair, Tensor_l_kpt_pair, DataType
from omnialigner.logging import logger as logging
from omnialigner.keypoints import KeypointPairs
from omnialigner.utils.field_transform import tfrs_to_grid_M, calculate_M_from_theta, tfrs_inv
from omnialigner.utils.point_transform import transform_keypoints

def _add_affine_transform(om_data: om.Omni3D, kd: KeypointPairs, i_layer: int, dense_dist_z: int):
    """Apply affine transformation to keypoint pairs.

    This internal function applies the affine transformation parameters from the
    affine alignment stage to keypoint pairs.

    Args:
        om_data (om.Omni3D): Omni3D object containing configuration and data.
        kd (KeypointPairs): Keypoint pairs object containing test points.
        i_layer (int): Current layer index.
        dense_dist_z (int): Z-distance for dense keypoint matching.

    Returns:
        tuple: Transformed keypoint coordinates (kpt_F_affine, kpt_M_affine),
        normalized by max_size.
    """
    kpt_F = kd["test_label"]
    kpt_M = kd["test_input"]
    
    dict_affine_model = torch.load(StageTag.AFFINE.get_file_name(om_data.proj_info)["affine_model"])
    key_F = f'grid2d_modules.{i_layer}.tensor_trs'
    key_M = f'grid2d_modules.{i_layer+dense_dist_z}.tensor_trs'
    if key_F not in dict_affine_model:
        key_F = f'{i_layer}.tensor_trs'
    if key_M not in dict_affine_model:
        key_M = f'{i_layer+dense_dist_z}.tensor_trs'

    tensor_tfrs_F = dict_affine_model[key_F]
    tensor_tfrs_M = dict_affine_model[key_M]
    grid_M_F = calculate_M_from_theta(tfrs_inv(tensor_tfrs_F), h=om_data.max_size, w=om_data.max_size)[0:2]
    grid_M_M = calculate_M_from_theta(tfrs_inv(tensor_tfrs_M), h=om_data.max_size, w=om_data.max_size)[0:2]

    kpt_F_affine = transform_keypoints(kpt_F, grid_M_F) / om_data.max_size
    kpt_M_affine = transform_keypoints(kpt_M, grid_M_M) / om_data.max_size
    return kpt_F_affine, kpt_M_affine


def _raw_to_affine(om_data: om.Omni3D, i_layer: int, kpt_raw: Tensor_kpts_N_xy):
    """Convert raw keypoint coordinates to affine-transformed coordinates.

    Args:
        om_data (om.Omni3D): Omni3D object containing configuration and data.
        i_layer (int): Layer index.
        kpt_raw (Tensor_kpts_N_xy): Raw keypoint coordinates.

    Returns:
        Tensor_kpts_N_xy: Affine-transformed keypoint coordinates.
    """
    kpt_F_affine = om.align.apply_affine_landmarks_HD(om_data, i_layer=i_layer, keypoints_raw=kpt_raw, zoom_level=0) 
    kpt_F_affine_out = kpt_F_affine.detach().cpu() * om_data.sizes[-1]
    return kpt_F_affine_out


def convert_to_raw_landmarks(om_data: om.Omni3D, i_layer: int, kpt_affined: Tensor_kpts_N_xy, zoom_level: float):
    """Convert affine-transformed landmarks back to raw coordinates.

    This function reverses the affine transformation and padding adjustments to
    convert landmarks back to their original coordinate space.

    Args:
        om_data (om.Omni3D): Omni3D object containing configuration and data.
        i_layer (int): Layer index.
        kpt_affined (Tensor_kpts_N_xy): Affine-transformed keypoint coordinates.
        zoom_level (float): Target zoom level for output coordinates.

    Returns:
        Tensor_kpts_N_xy: Raw landmark coordinates at specified zoom level.
    """
    tensor_tfrs_1 = om_data._load_TFRS_params(i_layer)
    dict_affine_model = torch.load(StageTag.AFFINE.get_file_name(om_data.proj_info)["affine_model"])
    key = f'grid2d_modules.{i_layer}.tensor_trs'
    if key not in dict_affine_model:
        key = f'{i_layer}.tensor_trs'

    tensor_tfrs_2 = dict_affine_model[key]

    grid_M_1 = calculate_M_from_theta(tfrs_to_grid_M(tensor_tfrs_1), h=om_data.max_size, w=om_data.max_size)[0:2]
    grid_M_2 = calculate_M_from_theta(tfrs_to_grid_M(tensor_tfrs_2), h=om_data.max_size, w=om_data.max_size)[0:2]

    lm_M_step1 = transform_keypoints(kpt_affined, grid_M_2)
    lm_pad = transform_keypoints(lm_M_step1, grid_M_1)
    pad_size_ = get_pad_size(om_data, i_layer=i_layer, zoom_level=om_data.zoom_level)
    pad_size = (pad_size_[1][1], pad_size_[1][0], pad_size_[0][1], pad_size_[0][0])

    scale_level = om_data.l_scales[zoom_level]
    lm_raw = lm_pad.clone()
    ratio = get_ratio(om_data, i_layer)
    lm_raw[:, 0] = (lm_raw[:, 0] - pad_size[0])*scale_level*ratio
    lm_raw[:, 1] = (lm_raw[:, 1] - pad_size[2])*scale_level*ratio
    return lm_raw


def calculate_dense_keypoints_pairs(
        om_data: om.Omni3D, 
        i_layer: int,
        dense_dist_z: int=1,
        dense_input_size: Tuple[int, int]=(1024, 1024),
        zoom_level: int=-1,
        overwrite_cache: bool=False,
        tag: str="raw",
        detector_roma=None
    ) -> Tensor_kpt_pair:
    """Calculate dense keypoint pairs between adjacent image layers.

    This function uses the ROMA detector to find dense correspondences between
    pairs of images. It handles both raw and processed images, and includes
    caching for efficiency.

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        i_layer (int): Index of the current layer.
        dense_dist_z (int, optional): Z-distance for matching layers. Defaults to 1.
        dense_input_size (Tuple[int, int], optional): Input size for ROMA detector.
            Defaults to (1024, 1024).
        zoom_level (int, optional): Target zoom level. Defaults to -1.
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.
        tag (str, optional): Data type tag ("raw", "gray", etc). Defaults to "raw".
        detector_roma (optional): Pre-initialized ROMA detector. Defaults to None.

    Returns:
        Tensor_kpt_pair: List containing fixed and moving keypoint coordinates
        [kpt_F_affine, kpt_M_affine].

    Note:
        - Results are cached by default for efficiency
        - Handles both raw and grayscale images
        - Automatically converts coordinates between different spaces
        - Generates visualization plots for verification
    """
    out_dir = f"{om_data.proj_info.root_dir}/analysis/{om_data.proj_info.project}/{om_data.proj_info.version}/04.detect_kpts/{om_data.proj_info.group}/roma_dense"
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/{i_layer}.pt"
    if dense_dist_z > 1:
        out_file = f"{out_dir}/{i_layer}_{dense_dist_z}.pt"
    if os.path.exists(out_file) and not overwrite_cache:
        kd = torch.load(out_file)
        kpt_F_affine = _raw_to_affine(om_data, i_layer, kd["test_label"] / om_data.sizes[zoom_level])
        kpt_M_affine = _raw_to_affine(om_data, i_layer+dense_dist_z, kd["test_input"] / om_data.sizes[zoom_level])
        return [kpt_F_affine / om_data.max_size, kpt_M_affine / om_data.max_size]

    if detector_roma is None:
        detector_roma = om.kp.init_detector("roma")
        detector_roma.set_upsample_res(dense_input_size)

    if i_layer+dense_dist_z >= len(om_data):
        return None

    om_data.set_tag(tag)
    is_raw = True
    if om_data.tag == DataType.GRAY:
        is_raw = False

    da_img_F = om.align.apply_affine_HD(om_data, i_layer=i_layer, tag=tag, overwrite_cache=True)
    da_img_M = om.align.apply_affine_HD(om_data, i_layer=i_layer+dense_dist_z, tag=tag, overwrite_cache=True)

    if not is_raw:
        da_img_F = da.repeat(da_img_F[:, :, 0:1], 3, axis=2)
        da_img_M = da.repeat(da_img_M[:, :, 0:1], 3, axis=2)

    kpt_F, kpt_M, matches, corresps = detector_roma.match(da_img_F, da_img_M)
    kpt_F = kpt_F.detach().cpu() / da_img_F.shape[0] * om_data.max_size
    kpt_M = kpt_M.detach().cpu() / da_img_M.shape[0] * om_data.max_size

    kpt_F_raw = convert_to_raw_landmarks(om_data, i_layer, kpt_F, zoom_level=0) * om_data.sizes[-1]
    kpt_M_raw = convert_to_raw_landmarks(om_data, i_layer+dense_dist_z, kpt_M, zoom_level=0) * om_data.sizes[-1]

    da_img_F = om_data.load_tiff(i_layer, is_raw=is_raw, zoom_level=om_data.zoom_level)
    da_img_M = om_data.load_tiff(i_layer+dense_dist_z, is_raw=is_raw, zoom_level=om_data.zoom_level)
    if not is_raw:
        da_img_F = np.repeat(da_img_F[:, :, 0:1].compute(), 3, axis=2)
        da_img_M = np.repeat(da_img_M[:, :, 0:1].compute(), 3, axis=2)

    img_F_raw = om.tl.im2tensor(da_img_F)
    img_M_raw = om.tl.im2tensor(da_img_M)

    kd_raw = KeypointPairs(image_F=img_F_raw, image_M=img_M_raw, mkpts_F=kpt_F_raw, mkpts_M=kpt_M_raw, index_matches=list(matches))
    if not os.path.exists(out_file) or overwrite_cache:
        fig = kd_raw.plot_dataset()
        os.makedirs(f"{out_dir}/figs", exist_ok=True)
        out_png = f"{out_dir}/figs/{i_layer}.png"
        if dense_dist_z > 1:
            out_png = f"{out_dir}/figs/{i_layer}_{dense_dist_z}.png"
        
        kd_raw.dataset['image_input'] = None
        kd_raw.dataset['image_label'] = None
        fig.savefig(out_png)
        torch.save(kd_raw.dataset, out_file, _use_new_zipfile_serialization=False)
    
    kpt_F_affine = _raw_to_affine(om_data, i_layer, kpt_F_raw / om_data.sizes[zoom_level])
    kpt_M_affine = _raw_to_affine(om_data, i_layer+dense_dist_z, kpt_M_raw / om_data.sizes[zoom_level])
    return [kpt_F_affine * om_data.sizes[zoom_level], kpt_M_affine * om_data.sizes[zoom_level]]


def calculate_dense_keypoints(
        om_data: om.Omni3D, 
        l_layers: List[int]=None,
        dense_dist_z: int=1,
        dense_input_size: Tuple[int, int]=(1024, 1024), 
        overwrite_cache: bool=False
    ) -> Tensor_l_kpt_pair:
    """Calculate dense keypoint pairs for multiple layers.

    This function processes multiple layers to generate dense keypoint correspondences
    between adjacent layers, with configurable z-distance.

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        l_layers (List[int], optional): List of layer indices to process.
            If None, processes all layers. Defaults to None.
        dense_dist_z (int, optional): Maximum z-distance for matching layers.
            Defaults to 1.
        dense_input_size (Tuple[int, int], optional): Input size for ROMA detector.
            Defaults to (1024, 1024).
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.

    Returns:
        Tensor_l_kpt_pair: List of lists containing keypoint pairs for each layer
        and z-distance combination.

    Note:
        - Uses a single ROMA detector instance for efficiency
        - Processes each layer with multiple z-distances up to dense_dist_z
        - Shows progress with tqdm
    """
    if l_layers is None:
        N_layers = len(om_data)
        l_layers = list(range(N_layers))

    detector_roma = om.kp.init_detector("roma")
    detector_roma.set_upsample_res(dense_input_size)
    l_kpt_pairs: Tensor_l_kpt_pair = []
    for i_layer in tqdm(l_layers, desc="Calculating dense keypoints"):
        layer_kpt_pairs: Tensor_Layer_kpt_pair = []
        for i_zdist in range(1, dense_dist_z+1):
            kpt_pairs = calculate_dense_keypoints_pairs(om_data, i_layer=i_layer, dense_dist_z=i_zdist, overwrite_cache=overwrite_cache, detector_roma=detector_roma)
            layer_kpt_pairs.append(kpt_pairs)
        l_kpt_pairs.append(layer_kpt_pairs)
    
    return l_kpt_pairs


def _add_dense_keypoints(kpts_moved: Tensor_kpts_N_xy, kpts_dense: Tensor_kpts_N_xy):
    """Combine moved keypoints with dense keypoints.

    Args:
        kpts_moved (Tensor_kpts_N_xy): Previously moved keypoints.
        kpts_dense (Tensor_kpts_N_xy): Dense keypoints to add.

    Returns:
        Tensor_kpts_N_xy: Combined keypoint coordinates.
    """
    if kpts_moved.shape[0] == 0:
        return kpts_dense
    
    kpt_out = torch.vstack([kpts_moved, kpts_dense])
    return kpt_out


def get_keypoints_pairs(
        om_data: om.Omni3D, 
        l_layers: List[int]=None, 
        l_kpt_pairs: Tensor_l_kpt_pair=None, 
        dense_dist_z: int=1, 
        overwrite_cache: bool=False
    ) -> Tensor_l_kpt_pair:
    """Get or calculate keypoint pairs for multiple layers.

    This function either uses provided keypoint pairs or calculates new ones
    using dense keypoint detection.

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        l_layers (List[int], optional): List of layer indices to process.
            If None, processes all layers. Defaults to None.
        l_kpt_pairs (Tensor_l_kpt_pair, optional): Pre-calculated keypoint pairs.
            If None, calculates new pairs. Defaults to None.
        dense_dist_z (int, optional): Maximum z-distance for matching layers.
            Defaults to 1.
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.

    Returns:
        Tensor_l_kpt_pair: List of lists containing keypoint pairs for each layer
        and z-distance combination.
    """
    if l_layers is None:
        N_layers = len(om_data)
        l_layers = range(N_layers)

    
    if l_kpt_pairs is None:
        l_kpt_pairs = [ [[ torch.tensor([]), torch.tensor([]) ]] for _ in l_layers ]
    
    l_kpt_pairs_ = []
    for idx in l_layers[:-1]:
        layer_kpt_pairs = l_kpt_pairs[idx]
        for i_zdist in range(1, dense_dist_z+1):
            kpts_dense = calculate_dense_keypoints_pairs(om_data, i_layer=idx, dense_dist_z=i_zdist, tag=om_data.tag, overwrite_cache=overwrite_cache)
            
            kpt_F = torch.tensor([])
            kpt_M = torch.tensor([])
            if kpts_dense is not None:
                kpt_F = kpts_dense[0]
                kpt_M = kpts_dense[1]

            if i_zdist == 1:
                kpt_F_ = layer_kpt_pairs[0][0]
                kpt_M_ = layer_kpt_pairs[0][1]
                kpt_F = _add_dense_keypoints(kpt_F_, kpt_F)
                kpt_M = _add_dense_keypoints(kpt_M_, kpt_M)
                layer_kpt_pairs = [[kpt_F, kpt_M]]
            else:
                layer_kpt_pairs.append([kpt_F, kpt_M])

        l_kpt_pairs_.append(layer_kpt_pairs)
    
    return l_kpt_pairs_

def nonrigid(
        om_data: om.Omni3D, 
        l_layers: List[int]=None,
        dense_dist_z: int=1,
        l_kpts_eval: List[Tensor_kpts_N_xy_raw|Tensor_kpt_pair]=None,
        overwrite_cache:bool=False
    ) -> Tuple[Tensor_image_NCHW, List[Tensor_kpts_N_xy_raw]]:
    """Apply non-rigid alignment to a sequence of images.

    This function performs non-rigid registration between adjacent image layers using
    dense keypoint correspondences. The process involves:
    1. Dense keypoint detection and matching
    2. Neural network-based optimization of deformation fields
    3. Application of learned transformations to images and keypoints

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        l_layers (List[int], optional): List of layer indices to process.
            If None, processes all layers. Defaults to None.
        dense_dist_z (int, optional): Maximum z-distance for matching layers.
            Defaults to 1.
        l_kpts_eval (List[Tensor_kpts_N_xy_raw|Tensor_kpt_pair], optional):
            Pre-computed keypoints for evaluation. Defaults to None.
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.

    Returns:
        tuple:
            - aligned_tensor (Tensor_image_NCHW): Non-rigidly aligned image tensor.
            - l_kpts_moved (List[Tensor_kpts_N_xy_raw]): List of transformed keypoints.

    Note:
        - Requires prior execution of affine alignment
        - Uses cached results when available
        - Supports evaluation with provided keypoints
        - Results are cached for future use
    """
    logging.info(f"Processing nonrigid, stage1, check inputs")
    FILE_cache_affine = StageTag.AFFINE
    dict_file_name = FILE_cache_affine.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name is None:
        logging.error("Affine model not found. Please run `om.align.affine()` first.")
        return None, None

    logging.info(f"Processing nonrigid, stage2, return cache if exists")
    dict_file_name_nonrigid = StageTag.NONRIGID.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name_nonrigid is not None and not overwrite_cache:
        l_kpts_moved = torch.load(dict_file_name_nonrigid["nonrigid_kpts"])
        aligned_tensor = torch.load(dict_file_name_nonrigid["padded_tensor"])
        if l_layers is not None:
            l_kpts_moved = [l_kpts_moved[i_layer] for i_layer in l_layers]
            aligned_tensor = aligned_tensor[l_layers, :, :, :]

        return aligned_tensor, l_kpts_moved
    
    logging.info(f"Processing nonrigid, stage3, expand l_layers")
    if l_layers is None:
        N_layers = len(om_data)
        l_layers = range(N_layers)
    else:
        overwrite_cache = False
    
    
    logging.info(f"Processing nonrigid, stage4, run")
    padded_tensors = torch.load(dict_file_name["padded_tensor"])[l_layers, :, :, :]
    l_kpt_pairs: Tensor_l_kpt_pair = torch.load(dict_file_name["affine_kpts"])
    dict_file_name_nonrigid = StageTag.NONRIGID.get_file_name(projInfo=om_data.proj_info, check_exist=False)
    out_dir_nonrigid = dict_file_name_nonrigid["out_dir"]
    os.makedirs(out_dir_nonrigid, exist_ok=True)
    l_kpt_pairs_ = get_keypoints_pairs(om_data=om_data, l_layers=l_layers, l_kpt_pairs=l_kpt_pairs, dense_dist_z=dense_dist_z, overwrite_cache=False)
    # torch.save(l_kpt_pairs_, "./l_kpt_pairs.pt")
    # return None, None
    i_pair = 0
    plt = om.pl.plt
    plt.scatter(l_kpt_pairs_[i_pair][0][0][:, 0], l_kpt_pairs_[i_pair][0][0][:, 1], s=1, c="red", label="F kpts")
    plt.scatter(l_kpt_pairs_[i_pair][0][1][:, 0], l_kpt_pairs_[i_pair][0][1][:, 1], s=1, c="blue", label="M kpts")
    model = OmniAligner(
        image_3d_tensor=padded_tensors, 
        l_kpt_pairs=l_kpt_pairs_, 
        log_prefix=out_dir_nonrigid, 
        dict_config=om_data.config["align"],
        model_type="nonrigid",
        save_prefix="nonrigid",
        full_figsize=om_data.plt_figsize,
        full_n_cols=om_data.plt_row_col[1]
    )
    n_epochs = om_data.config["align"]["nonrigid"].get("used_levels", 1)
    scale_level = om_data.max_size / om_data.sizes[om_data.zoom_level]
    l_kpts_eval_affine = None
    if l_kpts_eval is not None:
        l_kpts_eval_affine = []
        for i_layer, kpt_rawsize in enumerate(l_kpts_eval):
            if len(kpt_rawsize) == 1:
                kpt_rawsize = kpt_rawsize[0]
                kpt_affine = om.align.apply_affine_landmarks_HD(om_data, i_layer=i_layer, zoom_level=om_data.zoom_level, keypoints_raw=kpt_rawsize, unpad=False)
                kpt_affine = kpt_affine / (om_data.max_size * om_data.l_scales[om_data.zoom_level])
                l_kpts_eval_affine.append(kpt_affine)
            else:
                kpt_F_rawsize = kpt_rawsize[0]
                kpt_M_rawsize = kpt_rawsize[1]
                kpt_F_affine = om.align.apply_affine_landmarks_HD(om_data, i_layer=i_layer, zoom_level=om_data.zoom_level, keypoints_raw=kpt_F_rawsize, unpad=False)
                kpt_M_affine = om.align.apply_affine_landmarks_HD(om_data, i_layer=i_layer+1, zoom_level=om_data.zoom_level, keypoints_raw=kpt_M_rawsize, unpad=False)
                kpt_F_affine = kpt_F_affine / (om_data.max_size * om_data.l_scales[om_data.zoom_level])
                kpt_M_affine = kpt_M_affine / (om_data.max_size * om_data.l_scales[om_data.zoom_level])
                l_kpts_eval_affine.append([kpt_F_affine, kpt_M_affine])
    
    train_model(model, num_epochs=n_epochs, l_kpts_eval=l_kpts_eval_affine, scale_level=scale_level)
    logging.info("Align nonrigid done")

    model.eval()
    model.n_keypoints = -1
    aligned_tensor, l_kpts_moved_ = model.viz_all(model.image_3d_tensor, model.l_kpt_pairs)
    aligned_tensor = aligned_tensor.detach().cpu()
    l_kpts_moved = []
    for layer_kpts in l_kpts_moved_:
        layer_kpts_moved = []
        for kpt_pair in layer_kpts:
            kpt_F = kpt_pair[0].detach().cpu()
            kpt_M = kpt_pair[1].detach().cpu()
            layer_kpts_moved.append([kpt_F, kpt_M])
        l_kpts_moved.append(layer_kpts_moved)
    
    dict_model = OrderedDict()
    for idx in range(model.N):
        # dict_model[f"{idx}.displacement_field"] = model.grid2d_modules[idx].displacement_field
        dict_model[f"{idx}.displacement_field"] = model.grid2d_modules[idx].displacement_field
        
    logging.info(f"Processing nonrigid, stage5, save cache {overwrite_cache}")
    if overwrite_cache or not os.path.exists(dict_file_name_nonrigid["nonrigid_model"]):
        torch.save(aligned_tensor, dict_file_name_nonrigid["padded_tensor"], _use_new_zipfile_serialization=False)
        torch.save(l_kpts_moved, dict_file_name_nonrigid["nonrigid_kpts"], _use_new_zipfile_serialization=False)
        torch.save(dict_model, dict_file_name_nonrigid["nonrigid_model"], _use_new_zipfile_serialization=False)
    
    return aligned_tensor, l_kpts_moved




def warp_tensor(tensor: torch.Tensor, grid: torch.Tensor=None, displacement_field: torch.Tensor=None, mode: str='bilinear', padding_mode: str='zeros', device: str=None) -> torch.Tensor:
    """Warp a tensor using a grid or displacement field.

    This function applies spatial transformation to a tensor using either a sampling
    grid or a displacement field. It supports different interpolation modes and
    padding options.

    Args:
        tensor (torch.Tensor): Input tensor to warp [B, C, H, W].
        grid (torch.Tensor, optional): Sampling grid [B, H, W, 2]. Defaults to None.
        displacement_field (torch.Tensor, optional): Displacement field [B, 2, H, W].
            Defaults to None.
        mode (str, optional): Interpolation mode ('bilinear' or 'nearest').
            Defaults to 'bilinear'.
        padding_mode (str, optional): Padding mode ('zeros', 'border', or 'reflection').
            Defaults to 'zeros'.
        device (str, optional): Device to perform computation on. Defaults to None.

    Returns:
        torch.Tensor: Warped tensor with same shape as input.

    Note:
        Either grid or displacement_field must be provided, but not both.
        The function automatically handles device placement and grid generation.
    """
    if grid is None:
        grid = generate_grid(tensor=tensor, device=device)

    displacement_field = resample_displacement_field_to_size(displacement_field, (grid.size(1), grid.size(2)))
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return transformed_tensor


def warp_landmark(kpts: Tensor_kpts_N_xy, tensor_tfrs: torch.FloatTensor=torch.FloatTensor([0,0,0,0,0, 1,1]), displacement_field: torch.FloatTensor=None, tensor_size=None):
    """Warp landmark coordinates using transformation parameters.

    This function applies both affine and non-rigid transformations to landmark
    coordinates. It can handle both global affine transformations and local
    deformations specified by a displacement field.

    Args:
        kpts (Tensor_kpts_N_xy): Input landmark coordinates [N, 2].
        tensor_tfrs (torch.FloatTensor, optional): Affine transformation parameters
            [angle, tx, ty, sx, sy, 1, 1]. Defaults to identity transformation.
        displacement_field (torch.FloatTensor, optional): Displacement field for
            non-rigid deformation [2, H, W]. Defaults to None.
        tensor_size (tuple, optional): Size of the image tensor (H, W).
            Required if using displacement_field. Defaults to None.

    Returns:
        Tensor_kpts_N_xy: Transformed landmark coordinates.

    Note:
        - Affine transformation is applied first, followed by displacement field
        - Coordinates are normalized to [-1, 1] range during processing
        - For displacement field interpolation, coordinates must be within image bounds
    """
    grid_M = tfrs_to_grid_M(tensor_tfrs, device=kpts.device)
    grid = F.affine_grid(grid_M.unsqueeze(0), [1, 1, tensor_size[0], tensor_size[1]], align_corners=True)
        
    tensor_size_ = torch.tensor(tensor_size).to(kpts.device)
    kpts_new_scaled = None
    if kpts is not None:
        disp = displacement_field
        sampling_grid = (grid + disp)
        kpts_new_scaled = warp_landmark_grid(landmarks=kpts, grid=sampling_grid)
        kpts_new_scaled = kpts_new_scaled[:, 0, :]

    if tensor_size is None:
        _, H, W, _ = grid.shape
        tensor_size = torch.tensor(H, W).to(kpts.device)
    
    kpts_new_scaled = warp_landmark_grid(kpts, grid)
    return kpts_new_scaled


def resample_displacement_field_to_size(displacement_field: torch.Tensor, new_size: torch.Tensor, mode: str='bilinear') -> torch.Tensor:
    """Resample a displacement field to a new size.

    This function resizes a displacement field while preserving the relative
    displacements. The field values are scaled according to the size change.

    Args:
        displacement_field (torch.Tensor): Input displacement field [B, 2, H, W].
        new_size (torch.Tensor): Target size [H_new, W_new].
        mode (str, optional): Interpolation mode ('bilinear' or 'nearest').
            Defaults to 'bilinear'.

    Returns:
        torch.Tensor: Resampled displacement field [B, 2, H_new, W_new].

    Note:
        The displacement values are automatically scaled to maintain the same
        relative deformation effect at the new resolution.
    """
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), size=new_size, mode=mode, align_corners=False).permute(0, 2, 3, 1)


def warp_landmark_grid(landmarks: Tensor_kpts_N_xy, grid: torch.FloatTensor, k: int=1) -> Tensor_kpts_N_xy:
    """Warp landmarks using a sampling grid.

    This function transforms landmark coordinates using a provided sampling grid.
    It uses k-nearest neighbor interpolation to determine the displacement at
    each landmark location.

    Args:
        landmarks (Tensor_kpts_N_xy): Input landmark coordinates [N, 2].
        grid (torch.FloatTensor): Sampling grid [H, W, 2].
        k (int, optional): Number of nearest neighbors for interpolation.
            Defaults to 1.

    Returns:
        Tensor_kpts_N_xy: Transformed landmark coordinates.

    Note:
        - Coordinates are assumed to be in the range [0, grid_size]
        - Uses k-nearest neighbor interpolation for smooth transformations
        - Larger k values provide smoother but slower interpolation
    """
    _, H, W, _ = grid.shape
    source_coords = grid.view(-1, 2)
    target_coords = F.affine_grid(torch.eye(3)[:2, :].unsqueeze(0), (1, 1, H, W), align_corners=False).view(-1, 2).to(source_coords.device)
    
    tensor_coords = 2 * landmarks - 1
    batch_size = 1024  # Adjust batch size based on your memory constraints
    indices_list = []

    for i in range(0, tensor_coords.size(0), batch_size):
        batch_tensor_coords = tensor_coords[i:i + batch_size]
        distances = torch.cdist(batch_tensor_coords.unsqueeze(0), source_coords.unsqueeze(0)).squeeze(0)
        _, batch_indices = torch.topk(distances, k, largest=False)
        indices_list.append(batch_indices)

    indices = torch.cat(indices_list, dim=0)
    tensor_sources = source_coords[indices]
    tensor_target = 0.5 * (target_coords[indices] + 1)

    # keep grad in grid
    tensor_target = tensor_target + (tensor_sources - tensor_sources.detach())
    return tensor_target




def generate_grid(tensor: torch.Tensor=None, tensor_size: torch.Tensor=None, device: str=None) -> torch.Tensor:
    """Generate a regular sampling grid.

    Creates a regular grid of normalized coordinates in the range [-1, 1].
    The grid can be generated based on either a tensor's size or specified dimensions.

    Args:
        tensor (torch.Tensor, optional): Reference tensor for grid size.
            Defaults to None.
        tensor_size (torch.Tensor, optional): Explicit grid size [H, W].
            Defaults to None.
        device (str, optional): Device to create grid on. Defaults to None.

    Returns:
        torch.Tensor: Sampling grid [H, W, 2] with normalized coordinates.

    Note:
        - Either tensor or tensor_size must be provided
        - Grid coordinates are normalized to [-1, 1] range
        - Grid is compatible with torch.nn.functional.grid_sample
    """
    if tensor is not None:
        tensor_size = tensor.size()
    if device is None:
        identity_transform = torch.eye(len(tensor_size)-1)[:-1, :].unsqueeze(0).type_as(tensor)
    else:
        identity_transform = torch.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = torch.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def chunked_tile_grid_sample(input_tensor, grid, tile_size=(1024, 1024), N_chunk_size=1, C_chunk_size=1, align_corners=True):
    """Perform grid sampling in chunks for memory efficiency.

    This function applies grid sampling to large tensors by processing them in
    smaller chunks. This allows handling of very large images with limited memory.

    Args:
        input_tensor (torch.Tensor): Input tensor to sample [B, C, H, W].
        grid (torch.Tensor): Sampling grid [B, H, W, 2].
        tile_size (tuple, optional): Size of processing tiles (H, W).
            Defaults to (1024, 1024).
        N_chunk_size (int, optional): Batch dimension chunk size.
            Defaults to 1.
        C_chunk_size (int, optional): Channel dimension chunk size.
            Defaults to 1.
        align_corners (bool, optional): Grid sampling alignment mode.
            Defaults to True.

    Returns:
        torch.Tensor: Sampled tensor with same shape as input.

    Note:
        - Memory usage is proportional to tile_size * N_chunk_size * C_chunk_size
        - Larger chunk sizes increase memory usage but may improve performance
        - Results are identical to non-chunked grid_sample
        - Useful for processing high-resolution images with limited GPU memory
    """
    N, C, H, W = input_tensor.shape
    N_out, H_out, W_out, _ = grid.shape

    assert N == N_out, "Input tensor and grid must have the same batch size"
    idx_row, idx_col = max(H // tile_size[0], 1), max(W // tile_size[1], 1)
    
    output = torch.zeros(N, C, H_out, W_out)
    for i in range(idx_row):
        i_start = i * tile_size[0]
        i_end = min(i_start + tile_size[0], H_out)    
        for j in range(idx_col):
            j_start = j * tile_size[1]
            j_end = min(j_start + tile_size[1], W_out)
            grid_tile = grid[:, i_start:i_end, j_start:j_end, :]
            for n_start in range(0, N, N_chunk_size):
                n_end = min(n_start + N_chunk_size, N)
                
                grid_chunk = grid_tile[n_start:n_end]
                for c_start in range(0, C, C_chunk_size):
                    c_end = min(c_start + C_chunk_size, C)
                    input_chunk = input_tensor[n_start:n_end, c_start:c_end, :, :]
                    output_tile = F.grid_sample(input_chunk, grid_chunk, align_corners=align_corners)
                    output_tile = output_tile.cpu()
                    output[n_start:n_end, c_start:c_end, i_start:i_end, j_start:j_end] = output_tile
                    
    return output


if __name__ == '__main__':
    import omnialigner as om
    project = sys.argv[1]
    group = sys.argv[2]
    
    
    omni = om.Omni3D(config_info=f"/cluster/home/bqhu_jh/projects/omni/config/{project}/config_{group}.yaml")
    overwrite_cache = True
    om.align.nonrigid(omni, overwrite_cache=overwrite_cache)
    