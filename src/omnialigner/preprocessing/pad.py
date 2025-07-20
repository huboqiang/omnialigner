from typing import List, Tuple
import torch
from tqdm import tqdm
import dask.array as da
from dask_image.ndinterp import affine_transform
import numpy as np

import omnialigner as om
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageTag, StageSampleTag
from omnialigner.utils.image_pad import pad_tensors
from omnialigner.utils.image_viz import im2tensor, get_bg_tensor
from omnialigner.dtypes import Tensor_image_NCHW, Dask_image_HWC, DataType

def pad(om_data: om.Omni3D, l_layers:List[int]=None, overwrite_cache:bool=False) -> Tensor_image_NCHW:
    """Pad images to have uniform height and width dimensions.

    This function processes images to ensure they all have the same dimensions by padding
    or scaling as necessary. Results can be cached for future use.

    Args:
        om_data: Omni3D object containing image data and configuration.
        l_layers: Optional list of layer indices to process. If None, all layers are processed.
        overwrite_cache: Whether to overwrite existing cached results.

    Returns:
        Tensor_image_NCHW: Padded tensor containing all processed images.

    Raises:
        RuntimeError: If raw images for specified layers are not found.
    """
    # TODO, using dask instead of torch to save memory
    FILE_cache_sample_raw = StageSampleTag.RAW
    logging.info(f"Processing pad, stage1, check inputs")
    l_idxs = l_layers if l_layers is not None else range(len(om_data))
    for i_layer in l_idxs:
        dict_file_name = FILE_cache_sample_raw.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info)
        if dict_file_name is None:
            logging.error(f"Raw image for layer {i_layer} not found")
            return None

    logging.info(f"Processing pad, stage2, load cache")    
    FILE_cache_pad = StageTag.PAD
    dict_file_name = FILE_cache_pad.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name is not None and not overwrite_cache:
        padded_tensor = torch.load(dict_file_name["padded_tensor"])
        if l_layers is not None:
            return padded_tensor[l_layers, :, :, :]
        return padded_tensor

    logging.info(f"Processing pad, stage3, expand l_layers")
    if l_layers is None:
        l_layers = range(len(om_data))
    else:
        overwrite_cache = False
    
    logging.info(f"Processing pad, stage4, running")
    l_dask_NCHW = []
    l_bg_tensor = []
    for i_layer in tqdm(l_layers, desc="Loading tiff images"):
        dtype = om_data.proj_info.get_dtype(i_layer)
        is_bright = dtype == "HE"
        is_gray = om_data.tag == DataType.GRAY
        
        
        img = om_data.load_tiff(i_layer, zoom_level=om_data.zoom_level, is_raw=(not is_gray))
        l_dask_NCHW.append(img)

        bg_tensor = get_bg_tensor(N=1, is_bright=is_bright, is_gray=is_gray)
        if img.shape[2] == 1:
            bg_tensor = bg_tensor[:, 0:1, :, :]

        l_bg_tensor.append(bg_tensor)

    tensor_bg = torch.concat(l_bg_tensor)
    l_tensors = [ im2tensor( d.compute()) for d in l_dask_NCHW]

    l_padded_tensors, _, padded_sizes, l_ratio = pad_tensors(
        l_tensors, 
        [None for _ in range(len(l_tensors))], 
        max_size=[om_data.max_size, om_data.max_size], 
        pt_bg=tensor_bg
    )
    mean_ratio = torch.tensor(l_ratio).mean().item()
    if mean_ratio > 1.2:
        logging.warning(f"scaling ratio for fitting max_size is too high: {mean_ratio}. It is recommended to increase zoom_level or max_size in config file.")
    

    padded_tensors = torch.concat(l_padded_tensors)
    logging.info(f"Processing pad, stage5, save cache")
    if overwrite_cache:
        dict_file_name = FILE_cache_pad.get_file_name(projInfo=om_data.proj_info, check_exist=False)
        logging.info(f"Saving padded tensors to {dict_file_name}")
        padded_sizes = [ 
            (int(elem[0]/om_data.l_scales[om_data.zoom_level]),
             int(elem[1]/om_data.l_scales[om_data.zoom_level]), 
             int(elem[2]/om_data.l_scales[om_data.zoom_level]), 
             int(elem[3]/om_data.l_scales[om_data.zoom_level])) 
            for elem in padded_sizes
        ]
        torch.save(padded_tensors, dict_file_name["padded_tensor"], _use_new_zipfile_serialization=False)
        torch.save(padded_sizes, dict_file_name["merged_padded_sizes"], _use_new_zipfile_serialization=False)
        torch.save(l_ratio, dict_file_name["l_ratio"], _use_new_zipfile_serialization=False)

    return padded_tensors

def get_ratio(om_data: om.Omni3D, i_layer: int, **kwargs_tag) -> float:
    """Get the scaling ratio applied to a specific layer during padding.

    Args:
        om_data: Omni3D object containing project information.
        i_layer: Index of the layer to get ratio for.
        **kwargs_tag: Additional keyword arguments (unused).

    Returns:
        float: The scaling ratio applied to the specified layer.
    """
    ratio = torch.load(StageTag.PAD.get_file_name(om_data.proj_info)["l_ratio"])[i_layer]
    return ratio

def get_pad_size(om_data: om.Omni3D, i_layer: int, zoom_level:int=-1, **kwargs_tag) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Get the padding sizes applied to a specific layer at a given zoom level.

    Args:
        om_data: Omni3D object containing project information.
        i_layer: Index of the layer to get padding size for.
        zoom_level: Zoom level to calculate padding for. Defaults to -1.
        **kwargs_tag: Additional keyword arguments (unused).

    Returns:
        tuple: Triple of tuples containing padding sizes:
            - ((pad_right, pad_left), (pad_bottom, pad_top), (pad_channel_end, pad_channel_start))
    """
    scale_level = om_data.l_scales[zoom_level]
    pad_size = torch.load(StageTag.PAD.get_file_name(om_data.proj_info)["merged_padded_sizes"])[i_layer]
    pad_size_scaled = (
        (pad_size[3]*scale_level, pad_size[2]*scale_level), 
        (pad_size[1]*scale_level, pad_size[0]*scale_level), 
        (0, 0)
    )
    return pad_size_scaled


def _modify_image_size(da_image: Dask_image_HWC, target_size: Tuple[int, int]) -> Dask_image_HWC:
    """Modify image size through cropping or padding to match target dimensions.

    Args:
        da_image: Input dask array image in HWC format.
        target_size: Tuple of (target_height, target_width) dimensions.

    Returns:
        Dask_image_HWC: Modified image matching target dimensions.

    Note:
        If input image is larger, it will be cropped.
        If input image is smaller, it will be padded with zeros.
    """
    target_h, target_w = target_size
    height, width, channels = da_image.shape
    if height > target_h or width > target_w:
        da_image = da_image[:target_h, :target_w, :]
        return da_image
    
    pad_h = target_h - height
    pad_w = target_w - width
    pad_h_up = pad_h // 2
    pad_w_left = pad_w // 2
    da_image = da.pad(da_image, 
                    ((pad_h_up, pad_h-pad_h_up), (pad_w_left, pad_w-pad_w_left), (0, 0)), 
                    mode='constant', constant_values=0)
    return da_image

def input_adapter_for_align(om_data: om.Omni3D, i_layer: int, zoom_level=3, **kwargs_tag) -> Dask_image_HWC:
    """Adapt input image for alignment based on data type and configuration.

    This function handles different input types (SUB, CLS, RMAF, MASK, etc.) and
    prepares them for alignment processing.

    Args:
        om_data: Omni3D object containing image data and configuration.
        i_layer: Index of the layer to process.
        zoom_level: Zoom level for image loading. Defaults to 3.
        **kwargs_tag: Additional keyword arguments for specific data types.

    Returns:
        Dask_image_HWC: Processed image ready for alignment.
    """
    om_data.zoom_level = zoom_level
    
    if om_data.tag == DataType.SUB or om_data.tag == DataType.CLS:
        dict_src = om.pp.embed(om_data, i_layer=i_layer)
        tensor_src = dict_src[str(om_data.tag.name)]
        image_src = da.from_array(tensor_src[0].permute(1, 2, 0).detach().cpu().numpy())
    elif om_data.tag == DataType.RMAF:
        kwargs_rmaf = kwargs_tag.get("rmaf", {})
        image_src = om.pp.rmaf(om_data, i_layer=i_layer, **kwargs_rmaf)
    elif om_data.tag == DataType.MASK:
        for k in om_data.config["pp"]["crop"]:
            kwargs_tag[k] = om_data.config["pp"]["crop"][k]

        image_src = 1-om.pp.embed_mask(om_data, i_layer=i_layer, **kwargs_tag)
        image_src = da.from_array((255*image_src[:, :, None]).astype(np.uint8))
    else:
        # image_src, _ = read_image(om_data, i_layer=i_layer, zoom_level=zoom_level)
        is_raw = True
        if om_data.tag == DataType.GRAY:
            is_raw = False
        image_src = om_data.load_tiff(i_layer, is_raw=is_raw, zoom_level=zoom_level)

    return image_src
    

def pad_apply_to_omic_inputs(om_data: om.Omni3D, i_layer: int, zoom_level=3, **kwargs_tag) -> Dask_image_HWC:
    """Apply padding to omic inputs while maintaining alignment compatibility.

    This function handles padding for different data types (MASK, etc.) and ensures
    the results can be used with the same aligner.

    Args:
        om_data: Omni3D object containing image data and configuration.
        i_layer: Index of the layer to process.
        zoom_level: Zoom level for image processing. Defaults to 3.
        **kwargs_tag: Additional keyword arguments for mask calculation and padding.

    Returns:
        Dask_image_HWC: Padded and processed image.

    Note:
        For MASK type data, padding uses constant values of 0.
    """
    scale_level = om_data.l_scales[om_data.zoom_level] / om_data.l_scales[-1]
    if om_data.tag == DataType.MASK:
        kwargs_tag["mode"] = "constant"
        kwargs_tag["constant_values"] = 0

    image_src = input_adapter_for_align(om_data, i_layer, zoom_level=zoom_level, **kwargs_tag)
    if "pad_size" not in kwargs_tag:
        pad_size_scaled = get_pad_size(om_data, i_layer, zoom_level=zoom_level)
        kwargs_tag["pad_size"] = pad_size_scaled

    if "matrix" not in kwargs_tag:
        ratio = torch.load(StageTag.PAD.get_file_name(om_data.proj_info)["l_ratio"])[i_layer]
        matrix = np.array(
            [
                [ratio, 0, 0],
                [0, ratio, 0],
                [0, 0, 1],
            ]
        )
        kwargs_tag["matrix"] = matrix

    ky, kx = kwargs_tag["matrix"][0, 0], kwargs_tag["matrix"][1, 1]
    size_ = ( int(image_src.shape[0] / ky), int(image_src.shape[1] / kx))
    if "rmaf" in kwargs_tag:
        del kwargs_tag["rmaf"]

    da_zoomed = affine_transform(image_src, kwargs_tag["matrix"])[:size_[0], :size_[1], :]
    try:
        da_image_pad = da.pad(da_zoomed, **kwargs_tag)
    except:
        da_image_pad = da.pad(da_zoomed, pad_width=kwargs_tag["pad_size"], mode=kwargs_tag.get("mode", "constant"), constant_values=kwargs_tag.get("constant_values", 0))

    pad_size = (om_data.max_size*scale_level, om_data.max_size*scale_level)
    da_image_pad = _modify_image_size(da_image_pad, pad_size)
    return da_image_pad
