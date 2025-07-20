import os
from typing import Tuple

import dask.array as da
import torch
import numpy as np
from dask_image.ndinterp import affine_transform

import omnialigner as om
from omnialigner.preprocessing.pad import pad_apply_to_omic_inputs, get_pad_size
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageTag, StageSampleTag
from omnialigner.omni_3D import tag_decorator, zoom_level_decorator
from omnialigner.utils.image_transform import apply_tfrs_to_dask
from omnialigner.utils.point_transform import raw_landmarks_to_padded, transform_keypoints
from omnialigner.utils.field_transform import calculate_M_from_theta, tfrs_inv
from omnialigner.dtypes import Dask_image_HWC, Np_image_HWC, Tensor_kpts_N_xy_raw, Np_kpts_N_yx_raw, DataType, tag_map, Tensor_tfrs, Tensor_trs


def read_image_for_affine_HD(om_data: om.Omni3D, i_layer: int, **kwargs_tag) -> Dask_image_HWC:
    """Read and prepare high-definition image for affine transformation.

    This function reads the raw image data and applies necessary padding for HD affine
    transformation processing. It ensures the image is in the correct format and size
    for subsequent processing steps.

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        i_layer (int): Index of the layer to process.
        **kwargs_tag: Additional keyword arguments for padding configuration.
            Common options include:
            - mode: Padding mode (e.g., "constant")
            - constant_values: Value to use for constant padding

    Returns:
        Dask_image_HWC: Padded high-definition image in HWC format.

    Note:
        This function temporarily modifies the tag of the om_data object but
        restores it before returning.
    """
    raw_tag = om_data.tag
    
        
    da_image_pad = pad_apply_to_omic_inputs(om_data, i_layer=i_layer, zoom_level=om_data.zoom_level, **kwargs_tag)
    om_data.set_tag(raw_tag)
    return da_image_pad


def _affine_image_core(om_data: om.Omni3D, i_layer: int, **kwargs):
    """Core function for applying affine transformation to HD images.

    This internal function handles the main logic of applying affine transformations
    to high-definition images. It loads transformation parameters and applies them
    using dask for efficient processing of large images.

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        i_layer (int): Index of the layer to process.
        **kwargs: Additional arguments including:
            - tensor_tfrs_1: First transformation parameters
            - tensor_tfrs_2: Second transformation parameters

    Returns:
        Dask_image_HWC: Transformed high-definition image.

    Note:
        The function applies two sequential transformations:
        1. Padding/stacking transformation (tensor_tfrs_1)
        2. Affine alignment transformation (tensor_tfrs_2)
    """
    zoom_level = om_data.zoom_level
    sample_name = om_data.proj_info.get_sample_name(i_layer=i_layer)
    tensor_tfrs_1 = kwargs.get("tensor_tfrs_1", None)
    tensor_tfrs_2 = kwargs.get("tensor_tfrs_2", None)
    if tensor_tfrs_1 is None:
        tensor_tfrs_1 = om_data._load_TFRS_params(i_layer)

    if tensor_tfrs_2 is None:
        dict_affine_model = torch.load(StageTag.AFFINE.get_file_name(om_data.proj_info)["affine_model"])
        tag = f"grid2d_modules.{i_layer}.tensor_trs"
        if tag not in dict_affine_model:
            tag = f"{i_layer}.tensor_trs"

        tensor_tfrs_2 = dict_affine_model[tag]

    kwargs_pad = {"mode": "constant"}
    if om_data.proj_info.get_dtype(i_layer) == "HE" and om_data.tag == DataType.RAW:
        kwargs_pad["constant_values"] = 255

    if "rmaf" in kwargs:
        kwargs_pad["rmaf"] = kwargs["rmaf"]

    da_image_pad = read_image_for_affine_HD(om_data, i_layer=i_layer, **kwargs_pad)
    tile_size = om_data.config.get("pp", {}).get("vit_attention", {}).get("params", {}).get("tile_size", 2048)

    logging.info(f"Processing affine_HD, stage3, processing {sample_name} in zoom level {zoom_level}")
    da_affine_hd_HWC = apply_tfrs_to_dask(da_image_pad, [tensor_tfrs_1, tensor_tfrs_2], tile_size=(tile_size, tile_size, 1), **kwargs_pad)
    return da_affine_hd_HWC


def apply_affine_HD_custom(
        image: Dask_image_HWC|Np_image_HWC,
        pad_size: Tuple[int, int, int, int]=None,
        affine_matrix: np.ndarray=None,
        tensor_tfrs_1: Tensor_tfrs|Tensor_trs=None,
        tensor_tfrs_2: Tensor_tfrs|Tensor_trs=None,
        **kwargs
    ):
    if pad_size is None:
        pad_size = (0, 0, 0, 0)

    if affine_matrix is None:
        affine_matrix = np.eye(3)
    
    if tensor_tfrs_1 is None:
        tensor_tfrs_1 = torch.zeros(7).float()

    if tensor_tfrs_2 is None:
        tensor_tfrs_2 = torch.zeros(7).float()

    kwargs_pad = {}
    kwargs_pad["mode"] = kwargs.get("mode", "constant")
    kwargs_pad["constant_values"] = kwargs.get("constant_values", 0)

    tile_size = kwargs.get("tile_size", 2048)
    logging.info(f"Processing apply_affine_HD_custom, processing {image.shape} with {kwargs}")

    ky, kx = affine_matrix[0, 0], affine_matrix[1, 1]
    size_ = ( int(image.shape[0] // ky), int(image.shape[1] // kx))
    da_zoomed = affine_transform(image, affine_matrix)[:size_[0], :size_[1], :]
    da_image_pad = da.pad(da_zoomed, pad_size=pad_size, **kwargs_pad)
    da_affine_hd_HWC = apply_tfrs_to_dask(da_image_pad, [tensor_tfrs_1, tensor_tfrs_2], tile_size=(tile_size, tile_size, 1), **kwargs_pad)
    return da_affine_hd_HWC


@tag_decorator
def apply_affine_HD(om_data: om.Omni3D, i_layer: int, overwrite_cache:bool=False, tag:str|DataType=None, unpad:bool=False, **kwargs) -> Dask_image_HWC:
    """Apply affine transformation to high-definition images.

    This function applies affine transformation to full-resolution (RAW) images,
    handling the complexities of large image processing through dask arrays.
    The process includes:

    1. Image preparation:
       - Padding to match largest scale
       - Loading or computing transformation parameters
    2. Transformation application:
       - Combines padding/stacking and affine transformations
       - Uses dask for memory-efficient processing
    3. Cache management:
       - Stores results for future use
       - Handles different data types (RAW, SUB, CLS)

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        i_layer (int): Index of the layer to process.
        overwrite_cache (bool, optional): Whether to force recomputation and
            overwrite existing cache. Defaults to False.
        tag (str|DataType, optional): Data type tag to process. Defaults to None.
        unpad (bool, optional): Whether to remove padding after transformation.
            Defaults to False.
        **kwargs: Additional arguments passed to _affine_image_core.

    Returns:
        Dask_image_HWC: Transformed high-definition image in HWC format.

    Raises:
        RuntimeError: If required affine model is not found.

    Note:
        - This function can handle extremely large images (e.g., 51200x51200x3)
          with limited memory (e.g., 256GB) through dask's parallel processing.
        - Results are cached by default for efficiency in subsequent processing.
        - The transformation preserves the accuracy of torch.nn.functional.grid_sample
          while enabling processing of much larger images.
    """
    if isinstance(tag, str):
        tag = tag_map.get(tag.upper(), DataType.RAW)

    if tag is not None:
        om_data.set_tag(tag)

    zoom_level = om_data.zoom_level
    logging.info(f"Processing affine_HD, stage1, check inputs")
    FILE_cache_affine = StageTag.AFFINE
    dict_file_name = FILE_cache_affine.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name is None and not overwrite_cache:
        dict_file_name = FILE_cache_affine.get_file_name(projInfo=om_data.proj_info, check_exist=False)
        logging.error(f"Affine model {dict_file_name} not found. Please run `om.affine()` first.")
        return None


    logging.info(f"Processing affine_HD, stage2, return cache if exists, zoom_level={zoom_level}")
    FILE_cache_affine_HD = StageSampleTag.AFFINE_HD
    dict_file_name_affine_HD = FILE_cache_affine_HD.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)
    file_name_affine_HD = dict_file_name_affine_HD["zarr"]
    if zoom_level == 0 and os.path.exists(file_name_affine_HD) and not overwrite_cache and (om_data.tag!=DataType.SUB and om_data.tag!=DataType.CLS):
        da_affine_hd_HWC = da.from_zarr(file_name_affine_HD)
        if unpad:
            pad_size = get_pad_size(om_data, i_layer=i_layer, zoom_level=0)
            h, w, _ = da_affine_hd_HWC.shape
            da_affine_hd_HWC = da_affine_hd_HWC[pad_size[0][0]:h-pad_size[0][1], pad_size[1][0]:w-pad_size[1][1], :]
        
        return da_affine_hd_HWC
    

    
    da_affine_hd_HWC = _affine_image_core(om_data, i_layer, **kwargs)
    if unpad:
        tile_size = om_data.config.get("pp", {}).get("vit_attention", {}).get("params", {}).get("tile_size", 2048)
        pad_size = get_pad_size(om_data, i_layer=i_layer, zoom_level=zoom_level)

        h, w, _ = da_affine_hd_HWC.shape
        da_affine_hd_HWC = da_affine_hd_HWC[pad_size[0][0]:h-pad_size[0][1], pad_size[1][0]:w-pad_size[1][1], :]
        da_affine_hd_HWC = da.rechunk(da_affine_hd_HWC, chunks=(tile_size, tile_size, 1))
    
    if zoom_level == 0 and (overwrite_cache or not os.path.exists(file_name_affine_HD)) and om_data.tag == DataType.RAW:
        logging.info(f"Processing affine_HD, stage4, save to cache {file_name_affine_HD}")
        os.makedirs(os.path.dirname(file_name_affine_HD), exist_ok=True)
        da_affine_hd_HWC.to_zarr(file_name_affine_HD, overwrite=True)
    return da_affine_hd_HWC

@zoom_level_decorator
def apply_affine_landmarks_HD(om_data: om.Omni3D, i_layer: int, zoom_level: int, keypoints_raw: Tensor_kpts_N_xy_raw|Np_kpts_N_yx_raw, unpad:bool=False) -> Tensor_kpts_N_xy_raw:
    """Apply affine transformation to landmarks in high-definition images.

    This function transforms landmark coordinates using the same affine transformation
    parameters applied to the corresponding high-definition image. It handles the
    complexities of coordinate transformation across different zoom levels and padding
    configurations.

    The transformation process includes:
    1. Scale adjustment for zoom level
    2. Padding compensation
    3. Sequential application of two transformations:
       - Padding/stacking transformation
       - Affine alignment transformation
    4. Final coordinate adjustment for output zoom level

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        i_layer (int): Index of the layer to process.
        zoom_level (int): Target zoom level for output coordinates.
        keypoints_raw (Tensor_kpts_N_xy_raw|Np_kpts_N_yx_raw): Input landmarks in
            raw image coordinates (before padding).
        unpad (bool, optional): Whether to adjust coordinates for unpadded image.
            Defaults to False.

    Returns:
        Tensor_kpts_N_xy_raw: Transformed landmark coordinates scaled to the
        specified zoom level.

    Raises:
        RuntimeError: If required affine model is not found.

    Note:
        - The function handles both PyTorch tensor and NumPy array inputs
        - Coordinates are automatically scaled between zoom levels
        - Padding is handled automatically based on the image configuration
        - The transformation maintains consistency with the image transformation
          in apply_affine_HD
    """
    logging.info(f"Processing apply_affine_landmarks_HD, stage1, check inputs")
    om_data.set_zoom_level(zoom_level)

    FILE_cache_affine = StageTag.AFFINE
    dict_file_name = FILE_cache_affine.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name is None:
        dict_file_name = FILE_cache_affine.get_file_name(projInfo=om_data.proj_info, check_exist=False)
        logging.error(f"Affine model {dict_file_name} not found. Please run `om.affine()` first.")
        return None


    pad_size_ = get_pad_size(om_data, i_layer=i_layer, zoom_level=-1)
    pad_size = (pad_size_[1][1], pad_size_[1][0], pad_size_[0][1], pad_size_[0][0])

    ratio = torch.load(StageTag.PAD.get_file_name(om_data.proj_info)["l_ratio"])[i_layer]
    tensor_tfrs_1 = om_data._load_TFRS_params(i_layer)

    dict_affine_model = torch.load(StageTag.AFFINE.get_file_name(om_data.proj_info)["affine_model"])
    key = f'grid2d_modules.{i_layer}.tensor_trs'
    if key not in dict_affine_model:
        key = f'{i_layer}.tensor_trs'

    tensor_tfrs_2 = dict_affine_model[key]
    if isinstance(keypoints_raw, Np_kpts_N_yx_raw):
        keypoints_raw = torch.from_numpy(keypoints_raw).float()

    ### om.pp.pad -> om.align.stack -> om.align.affine were calculated in zoom_level -1
    kpt_zoom_level_smallest = keypoints_raw / (om_data.l_scales[zoom_level] / om_data.l_scales[-1])
    lm_pad = raw_landmarks_to_padded(kpt_zoom_level_smallest, ratio=1/ratio, padded_size=pad_size)
    grid_M_1 = calculate_M_from_theta(tfrs_inv(tensor_tfrs_1), h=om_data.max_size, w=om_data.max_size)[0:2]
    grid_M_2 = calculate_M_from_theta(tfrs_inv(tensor_tfrs_2), h=om_data.max_size, w=om_data.max_size)[0:2]
    lm_M_step1 = transform_keypoints(lm_pad, grid_M_1)
    lm_M_step2 = transform_keypoints(lm_M_step1, grid_M_2)
    ### upsample to the current zoom level
    lm_M_out = lm_M_step2 * om_data.l_scales[om_data.zoom_level] / om_data.l_scales[-1]
    if unpad:
        pad_size = get_pad_size(om_data, i_layer=i_layer, zoom_level=om_data.zoom_level)
        lm_M_out[:, 0] -= pad_size[1][0]
        lm_M_out[:, 1] -= pad_size[0][0]

    return lm_M_out
