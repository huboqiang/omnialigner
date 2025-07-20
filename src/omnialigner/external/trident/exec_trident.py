import os
import subprocess
from typing import Tuple

import scanpy as sc
import numpy as np

import omnialigner as om
from omnialigner.cache_files import StageSampleTag
from omnialigner.utils.sd_zarr import load_spatial_adata_from_h5

exec_trident = os.path.join(os.path.dirname(__file__), "../../vendor/TRIDENT/run_single_slide.py")

def crop_adata_coords(om_data: om.Omni3D, adata: sc.AnnData, i_layer: int=0, target_crop_zoomlevel: int=1) -> np.ndarray:
    da_raw = om_data.load_tiff(i_layer, zoom_level=0)
    h, w, _ = da_raw.shape

    dir_crop = StageSampleTag.RAW.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)["dir"] + "/crop"
    sample_name = om_data.proj_info.get_sample_name(i_layer)
    coords = adata.obsm["spatial"]
    np_coords = [0, 0, 1, 1]
    if os.path.exists(f"{dir_crop}/{sample_name}_coords.npy"):
        np_coords = np.load(f"{dir_crop}/{sample_name}_coords.npy")

    h_crop = h / (np_coords[2]-np_coords[0]) * np_coords[0]
    w_crop = w / (np_coords[3]-np_coords[1]) * np_coords[1]
    ratio = om_data.l_scales[target_crop_zoomlevel] / om_data.l_scales[0]
    coords_cropped = np.stack([ (coords[:, 0]-w_crop)*ratio , (coords[:, 1]-h_crop)*ratio], axis=1)
    return coords_cropped


def run_trident(om_data: om.Omni3D, i_layer: int=0, overwrite_cache=False, model="conch_v15", patch_size=256, mag=20, crop=False, target_crop_zoomlevel=1) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Run TRIDENT on the given slide data.
    This function will execute the TRIDENT algorithm on the specified slide data and return adata objects
    
    Args:
        om_data (Omni3D): The Omni3D object containing project configuration and data.
        zoom_level (int, optional): The zoom level to use for image alignment and output. Defaults to 1.
        model (str, optional): [conch_v15, conch_v1, uni_v1, gigapath, vichow]. Defaults to "conch_v15".
        patch_size (int, optional): patch size for trident. Defaults to 256.
        mag (int, optional): mat for trident. Defaults to 20.

    Returns:
        Tuple[sc.AnnData, sc.AnnData]: AnnData objects for cls[N, C] and attention[(N, h, w), C].
    """
    dict_file_RAW = StageSampleTag.DATA.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)
    slide = dict_file_RAW["data"]

    dict_file_TRIDENT = StageSampleTag.TRIDENT_DATA.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)
    dir_TRIDENT = dict_file_TRIDENT["dir"]
    cls_h5 = dict_file_TRIDENT["cls"]
    sub_h5 = dict_file_TRIDENT["sub"]
    print(cls_h5, sub_h5)
    if overwrite_cache or not os.path.exists(sub_h5):
        os.makedirs(dir_TRIDENT, exist_ok=True)
        # Run TRIDENT
        cmds = ["python",
                        exec_trident,
                        "--slide_path", slide,
                        "--job_dir", dir_TRIDENT,
                        "--patch_encoder", model,
                        "--mag", f"{mag}",
                        "--patch_size", f"{patch_size}"]
        print(cmds)
        subprocess.run(cmds, shell=True, check=True)

    print("converting to AnnData")
    adata_cls = load_spatial_adata_from_h5(cls_h5)
    adata_sub = load_spatial_adata_from_h5(sub_h5)
    if crop:
        adata_cls.obsm["spatial"] = crop_adata_coords(om_data, adata_cls, i_layer=i_layer, target_crop_zoomlevel=target_crop_zoomlevel)
        adata_sub.obsm["spatial"] = crop_adata_coords(om_data, adata_sub, i_layer=i_layer, target_crop_zoomlevel=target_crop_zoomlevel)
    
    return adata_cls, adata_sub
