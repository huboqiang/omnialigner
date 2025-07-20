
import os
from typing import Tuple

import cv2
import dask.array as da
import numpy as np

from omnialigner.preprocessing.crop import crop
from omnialigner.preprocessing.dino import calculate_crop_margin_dino
from omnialigner.logging import logger as logging

from omnialigner.cache_files import StageSampleTag
from omnialigner.omni_3D import Omni3D
from omnialigner.datasets.datasets import read_file
from omnialigner.utils.io import write_qptiff_2d
from omnialigner.dtypes import DataType

def read_image(
        omni: Omni3D, 
        i_layer:int=0,
        zoom_level:int=-1,
        crop_image:bool=False,
        overwrite_cache:bool=False
    ) -> Tuple[da.Array, np.ndarray]:
    """
    Read image file and crop it if needed.
    From input image into "RAW" stage
    
    Args:
        file_name: str, path to the input file. png or tiff.
        is_tiff: bool, whether the file is tiff
        crop_image: bool, whether to crop the image
        kwargs: dict, additional arguments.
                "datasets" is required if read as tiff
                "crop" is required if crop the imagetorch.threshold

    Returns:
        da.Array, da.array of the image
        np_coords: np.ndarray, 4x1 array, scaled to [0, 1], [i_beg, j_beg, i_end, j_end]
    """
    logging.info(f"Processing read_image, stage1, load cache")    
    FILE_cache_sample_raw = StageSampleTag.RAW
    dict_file_name = FILE_cache_sample_raw.get_file_name(i_layer=i_layer, projInfo=omni.proj_info)
    omni.set_tag(omni.tag)
    is_raw = (omni.tag == DataType.RAW)
    kwargs_proj = {"project": omni.project, "group": omni.group}
    if dict_file_name is not None and not overwrite_cache:
        da_arr = read_file(
            dict_file_name["raw"],
            is_tiff=True,
            zoom_level=zoom_level,
            is_raw=is_raw,
            resize_to_20x=False,
            sizes=omni.sizes,
            **kwargs_proj
        )
        return da_arr, None


    FILE_cache_sample_DATA = StageSampleTag.DATA
    logging.info(f"Processing read_image, stage2, check inputs")
    
    dict_file_name = FILE_cache_sample_DATA.get_file_name(i_layer=i_layer, projInfo=omni.proj_info)
    if dict_file_name is None:
        logging.error(f"Raw image for layer {i_layer} not found")
        return None

    infile = dict_file_name["data"]
    sample_name = omni.proj_info.get_sample_name(i_layer)

    
    logging.info(f"Processing read_image, stage3, expand l_layers")
    
    logging.info(f"Processing read_image, stage4, read file")
    dict_file_name = FILE_cache_sample_raw.get_file_name(i_layer=i_layer, projInfo=omni.proj_info, check_exist=False)
    out_file = dict_file_name["raw"]
    out_dir = os.path.dirname(out_file)
    os.makedirs(f"{out_dir}/crop", exist_ok=True)
    
    
    da_arr = read_file(infile, is_tiff=omni.raw_is_tiff, zoom_level=zoom_level, is_raw=True, resize_to_20x=False, sizes=omni.sizes, **kwargs_proj)
    
    np_coords = None
    if crop_image:
        zoom_level_ = -1
        if infile.split(".")[-1] == "qptiff":
            zoom_level_ = -2
        
        da_arr_ = read_file(infile, is_tiff=omni.raw_is_tiff, zoom_level=zoom_level_, is_raw=True, resize_to_20x=False, sizes=omni.sizes, **kwargs_proj)
        kwargs_crop = omni.config["pp"]["crop"]
        if da_arr_.shape[2] != 3:
            da_arr_ = da.repeat(da_arr_[:, :, -1:], 3, axis=2)
        
        np_mask, np_coords, _ = calculate_crop_margin_dino(da_arr_, **kwargs_crop)
        logging.info(f"crop {infile} with {np_coords} to {out_dir}")
        if overwrite_cache:
            cv2.imwrite(f"{out_dir}/crop/{sample_name}_mask.png", np_mask)
            np.save(f"{out_dir}/crop/{sample_name}_coords.npy", np_coords)

        da_arr = crop(da_arr, np_coords)

    sizes = (1 / (np.array(omni.l_scales) / 40)).astype(int).tolist()
    write_qptiff_2d(out_file, da_arr.compute(), sizes=sizes)

    return da_arr, np_coords

