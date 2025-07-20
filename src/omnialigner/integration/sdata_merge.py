import os
from typing import Tuple

import numpy as np
import spatialdata as sd
import torch

import omnialigner as om
from omnialigner.cache_files import StageTag
from omnialigner.utils.sd_zarr import merge_spatialdata_with_scaling, write_omnialigner_zarr

def join_sdata_with_zs(om_data: om.Omni3D, zoom_level: int=1, overwrite_cache=False) -> sd.SpatialData:
    """
    This function merged aligned omni-aligner sd results with lazyslide sd results in zoom_level 1.

    Args:
        om_data (Omni3D): Description of om_data.
        zoom_level (int, optional): Description of zoom_level. Defaults to 1.

    Returns:
        SpatialData: Description of the return value.

    Raises:
        Exception: Description of the exception raised (if any).

    """
    sdata = write_omnialigner_zarr(om_data, zoom_level=1, overwrite_cache=overwrite_cache)
    sdata1 = om.external.run_lazy_slide(om_data, i_layer=0, overwrite_cache=overwrite_cache)
    xoff, yoff, scale_factor = calculate_raw_offset(om_data, zoom_level=zoom_level)
    sdata_merged = merge_spatialdata_with_scaling(sdata, sdata1, scale_factor=scale_factor, xoff=xoff, yoff=yoff)
    return sdata_merged


def calculate_raw_offset(om_data: om.Omni3D, zoom_level: int=1) -> Tuple[float, float, float]:
    """_summary_

    Args:
        om_data (om.Omni3D): _description_
        zoom_level (int, optional): _description_. Defaults to 1.

    Returns:
        Tuple[float, float, float]: _description_
    """
    root_dir = os.path.expanduser(om_data.config["datasets"]["root_dir"])
    project = om_data.config["datasets"]["project"]
    group_id = om_data.config["datasets"]["group"]
    version = om_data.config["datasets"]["version"]
    scale_factor = om_data.l_scales[zoom_level] / om_data.l_scales[0]
    da_zoom1 = om_data.load_tiff(0, zoom_level=zoom_level)
    h, w, _ = da_zoom1.shape
    np_coords = np.load(f"{root_dir}/analysis/{project}/{version}/01.ome_tiff/{group_id}/crop/HE_coords.npy")
    pad_size = om.pp.pad_size(om_data, i_layer=0, zoom_level=zoom_level)
    h_crop = h / (np_coords[2]-np_coords[0]) * np_coords[0]
    w_crop = w / (np_coords[3]-np_coords[1]) * np_coords[1]
    h_pad = pad_size[0][0]
    w_pad = pad_size[1][0]
    xoff = -w_crop + w_pad
    yoff = -h_crop + h_pad

    dict_pad = StageTag.PAD.get_file_name(om_data.proj_info, check_exist=False)
    ratio = torch.load(dict_pad["l_ratio"])[0]
    return xoff/ratio, yoff/ratio, scale_factor/ratio
