import os

import dask.array as da
import torch
import torch.nn.functional as F

import omnialigner as om
from omnialigner.utils.point_transform import warp_landmark_grid_faiss as warp_landmark_grid
from omnialigner.utils.field_transform import tfrs_to_grid_M, resample_displacement_field_to_size
from omnialigner.align.models.grid_2d.deeperhistreg_module import warp_tensor
from omnialigner.cache_files import StageTag, StageSampleTag
from omnialigner.dtypes import Grid2DModelDual, Tensor_kpts_N_xy_raw, Np_kpts_N_yx_raw, Dask_image_HWC, DataType, tag_map
from omnialigner.logging import logger as logging
from omnialigner.omni_3D import tag_decorator, zoom_level_decorator
from omnialigner.utils.config import load_from_string


def init_nonrigid_model(om_data: om.Omni3D, i_layer: int) -> Grid2DModelDual:
    """Initialize non-rigid transformation model for a specific layer.

    This function loads and initializes a non-rigid transformation model with
    pre-trained parameters for a given layer. The model is configured according
    to the project settings and scaled to match the current zoom level.

    Args:
        om_data (om.Omni3D): Omni3D object containing configuration and data.
        i_layer (int): Layer index to initialize model for.

    Returns:
        Grid2DModelDual: Initialized non-rigid transformation model.

    Note:
        - Model parameters are loaded from cached files
        - Model size is adjusted based on current zoom level
        - Configuration is taken from om_data.config["align"]["nonrigid"]
    """
    dict_config = om_data.config["align"]["nonrigid"]
    model_name = dict_config["model"]

    for k in ["initial_displacement_field", "tensor_size"]:
        if k in dict_config:
            del dict_config[k]

    file_grid_model = StageTag.NONRIGID.get_file_name(om_data.proj_info)["nonrigid_model"]
    dict_params = torch.load(file_grid_model, map_location="cpu")
    max_size = om_data.max_size * om_data.l_scales[om_data.zoom_level]
    return load_from_string(model_name)(
        initial_displacement_field=dict_params[f"{i_layer}.displacement_field"],
        tensor_size=(max_size, max_size),
        **dict_config
    )

@tag_decorator
def apply_nonrigid_HD(
            om_data: om.Omni3D,
            i_layer: int,
            tag: DataType|str="raw",
            overwrite_cache: bool=False,
            unpad:bool=False,
            **kwargs
    ) -> Dask_image_HWC:
    """Apply non-rigid transformation to high-definition images.

    This function applies non-rigid transformation to high-resolution images,
    handling different data types (raw, embedded, classified) and zoom levels.
    It includes caching mechanisms for efficient processing of large images.

    The transformation process involves:
    1. Loading or computing affine-transformed image
    2. Applying non-rigid deformation field
    3. Handling padding and chunking for memory efficiency
    4. Caching results for future use

    Args:
        om_data (om.Omni3D): Omni3D object containing image data and configuration.
        i_layer (int): Layer index to process.
        tag (DataType|str, optional): Data type tag ("raw", "sub", "cls", etc).
            Defaults to "raw".
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.
        unpad (bool, optional): Whether to remove padding after transformation.
            Defaults to False.
        **kwargs: Additional arguments passed to transformation functions.

    Returns:
        Dask_image_HWC: Transformed high-definition image.

    Note:
        - Supports different data types with appropriate processing
        - Uses dask arrays for memory-efficient processing
        - Results are cached based on data type and zoom level
        - Handles both uint8 and float data types
    """
    if isinstance(tag, str):
        tag = tag_map.get(tag.upper(), DataType.RAW)

    dict_params = StageTag.NONRIGID.get_file_name(om_data.proj_info)
    zoom_level = om_data.zoom_level
    if dict_params is None:
        logging.error("Nonrigid model not found. Please run `om.align.nonrigid()` first.")
        return None

    if tag == DataType.SUB or tag == DataType.CLS:
        file_embed = StageSampleTag.EMBED_NONRIGID.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info, check_exist=False)["embed_nonrigid"]
        if os.path.exists(file_embed) and not overwrite_cache:
            logging.info(f"Loading nonrigid transformation of embedding from {file_embed}")
            da_embed = da.from_zarr(file_embed)
            return da_embed
    else:
        expected_zoom_level = 1 if tag == DataType.RMAF else 0
        file_nonrigid = StageSampleTag.NONRIGID_HD.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info, check_exist=False)["zarr"]
        if zoom_level == expected_zoom_level and os.path.exists(file_nonrigid) and not overwrite_cache:
            logging.info(f"Loading nonrigid transformation of raw image from {file_nonrigid}")
            da_nonrigid = da.from_zarr(file_nonrigid)
            return da_nonrigid

    da_affine = om.align.apply_affine_HD(om_data, i_layer=i_layer, tag=tag, overwrite_cache=True, **kwargs)
    chunk_size = da_affine.chunksize
    grid_model: Grid2DModelDual = init_nonrigid_model(om_data, i_layer)
    device = torch.device("cpu")
    grid_model.set_device(device)
    logging.info(f"Applying nonrigid transformation to layer {i_layer} in zoom level {zoom_level}")
    with torch.no_grad():
        is_uint8 = True
        if tag == DataType.SUB or tag == DataType.CLS:
            is_uint8 = False

        tensor_affine = om.tl.im2tensor(da_affine, is_uint8=is_uint8)
        disp = grid_model.disp
        kwargs = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}
        if tag == DataType.SUB or tag == DataType.CLS:
            kwargs = {"mode": "nearest", "padding_mode": "zeros"}

        tensor_nonrigid = warp_tensor(tensor_affine, disp, **kwargs)
        np_nonrigid = om.tl.tensor2im(tensor_nonrigid, is_uint8=is_uint8)

    da_nonrigid = da.from_array(np_nonrigid, chunks=chunk_size)
    if unpad:
        tile_size = om_data.config.get("pp", {}).get("vit_attention", {}).get("params", {}).get("tile_size", 2048)
        pad_size = om.pp.pad_size(om_data, i_layer=i_layer, zoom_level=zoom_level)
        h, w, _ = da_nonrigid.shape
        print("pad_size:", pad_size)
        da_nonrigid = da_nonrigid[pad_size[0][0]:h-pad_size[0][1], pad_size[1][0]:w-pad_size[1][1], :]
        da_nonrigid = da.rechunk(da_nonrigid, chunks=(tile_size, tile_size, 1))

    if (tag == DataType.SUB or tag == DataType.CLS):
        if overwrite_cache or not os.path.exists(file_embed):
            da_nonrigid.to_zarr(file_embed, overwrite=True)
    elif tag == DataType.RMAF:
        if zoom_level == 1 and (overwrite_cache or not os.path.exists(file_nonrigid)):
            da_nonrigid.to_zarr(file_nonrigid, overwrite=True)
    else:
        if zoom_level == 0 and (overwrite_cache or not os.path.exists(file_nonrigid)):
            da_nonrigid.to_zarr(file_nonrigid, overwrite=True)

    return da_nonrigid

@zoom_level_decorator
def apply_nonrigid_landmarks_HD(om_data: om.Omni3D, i_layer: int, zoom_level: int, keypoints_raw: Tensor_kpts_N_xy_raw|Np_kpts_N_yx_raw, unpad:bool=False) -> Tensor_kpts_N_xy_raw:
    """Apply non-rigid transformation to landmark coordinates in high-definition images.

    This function transforms landmark coordinates using the same non-rigid deformation
    field applied to the corresponding high-definition image. It handles both
    PyTorch tensors and NumPy arrays as input.

    The transformation process includes:
    1. Converting landmarks to affine-transformed space
    2. Applying non-rigid deformation field
    3. Scaling coordinates to target zoom level
    4. Optional padding removal

    Args:
        om_data (om.Omni3D): Omni3D object containing configuration and data.
        i_layer (int): Layer index to process.
        zoom_level (int): Target zoom level for output coordinates.
        keypoints_raw (Tensor_kpts_N_xy_raw|Np_kpts_N_yx_raw): Input landmarks in
            raw image coordinates (before padding).
        unpad (bool, optional): Whether to adjust coordinates for unpadded image.
            Defaults to False.

    Returns:
        Tensor_kpts_N_xy_raw: Transformed landmark coordinates scaled to the
        specified zoom level.

    Note:
        - Requires prior execution of non-rigid alignment
        - Handles both PyTorch tensor and NumPy array inputs
        - Coordinates are automatically scaled between zoom levels
        - Maintains consistency with image transformation in apply_nonrigid_HD
    """
    dict_params = StageTag.NONRIGID.get_file_name(om_data.proj_info)
    if dict_params is None:
        logging.error("Nonrigid model not found. Please run `om.align.nonrigid()` first.")
        return None

    if isinstance(keypoints_raw, Np_kpts_N_yx_raw):
        keypoints_raw = torch.from_numpy(keypoints_raw).float()

    kpt_affine = om.align.apply_affine_landmarks_HD(om_data, i_layer=i_layer, zoom_level=zoom_level, keypoints_raw=keypoints_raw, unpad=False)
    grid_model: Grid2DModelDual = init_nonrigid_model(om_data, i_layer)
    device = torch.device("cpu")
    grid_model.set_device(device)
    max_size = int(om_data.max_size * om_data.l_scales[om_data.zoom_level] / om_data.l_scales[-1])
    with torch.no_grad():
        grid_model = init_nonrigid_model(om_data, i_layer=i_layer)
        grid_model.set_device(torch.device("cpu"))
        grid_M = tfrs_to_grid_M(grid_model.tensor_trs, device=grid_model.dev)
        grid = F.affine_grid(grid_M.unsqueeze(0), [1, 1, max_size, max_size], align_corners=True)
        displacement_field_smoothed = grid_model.disp
        displacement_field_smoothed_ = resample_displacement_field_to_size(displacement_field_smoothed, (max_size, max_size))
        kpts_new_scaled = warp_landmark_grid(kpt_affine / max_size, grid=grid+displacement_field_smoothed_)
        out_kpt = kpts_new_scaled[:, 0, :]

        out_kpt = out_kpt * max_size

    if unpad:
        pad_size = om.pp.pad_size(om_data, i_layer=i_layer, zoom_level=om_data.zoom_level)
        out_kpt[:, 0] -= pad_size[1][0]
        out_kpt[:, 1] -= pad_size[0][0]

    return out_kpt
