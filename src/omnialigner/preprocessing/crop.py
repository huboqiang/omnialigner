import os

import dask.array as da
import numpy as np
import cv2

import omnialigner as om
from omnialigner.cache_files import StageSampleTag
from omnialigner.datasets.datasets import read_file
from omnialigner.preprocessing.dino import calculate_crop_margin_dino
from omnialigner.utils.image_viz import rgb_he_to_gray
from omnialigner.utils.io import write_qptiff_2d

def crop(
        da_arr: da.Array, 
        np_coords: np.ndarray=None
    ) -> da.Array:
    """Crop input image with given coords
        
        Args:
            da_arr: HWC da.Array , input image
            np_coords: np.ndarray, 4x1 array, scaled to [0, 1], [i_beg, j_beg, i_end, j_end]
        
        Returns:
            da.Array, HWC da.Array of the cropped image
    """
    if np_coords is None:
        np_coords = np.array([0, 0, 1, 1])
    
    H, W, _ = da_arr.shape

    i_beg = int(np_coords[0]*H)
    j_beg = int(np_coords[1]*W)
    i_end = int(np_coords[2]*H)
    j_end = int(np_coords[3]*W)
    
    return da_arr[i_beg:i_end, j_beg:j_end, :]


def crop_image(
        om_data: om.Omni3D,
        i_layer: int,
        crop_image=True,
        overwrite_cache=False,
        zoom_level_crop_bg=-1,
        **kwargs_proj
    ) -> da.Array:
    """
        Crop input image using embedding of ViT.

        Args:
            om_data: Omni3D class.
            i_layer: index of layer for crop.
            crop_image: if crop raw image accroding to embedding.
            overwrite_cache: Whether to overwrite the cache. Notice if `l_layers` is not None, overwrite_cache will be forced into false
        
        Returns:
            da.Array, HWC da.Array of the cropped image
    """
    sample_name = om_data.proj_info.get_sample_name(i_layer)
    dict_file_name = StageSampleTag.DATA.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info)
    infile = dict_file_name["data"]

    da_arr = read_file(infile, is_tiff=om_data.raw_is_tiff, zoom_level=om_data.zoom_level, is_raw=True, resize_to_20x=False, sizes=om_data.sizes, **kwargs_proj)
    
    dict_file_name = StageSampleTag.RAW.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info, check_exist=False)
    out_file = dict_file_name["raw"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    os.makedirs(f"{os.path.dirname(out_file)}/crop/", exist_ok=True)

    sizes = (1 / (np.array(om_data.l_scales) / 40)).astype(int).tolist()
    if not crop_image:
       if overwrite_cache:
           write_qptiff_2d(out_file, da_arr.compute(), sizes=sizes)
       return da_arr

    np_coords = get_crop_coords(om_data, i_layer, zoom_level_crop_bg=zoom_level_crop_bg, overwrite_cache=overwrite_cache, **kwargs_proj)
    om_data.set_zoom_level(0)
    kwargs_proj["project"] = om_data.proj_info.project
    kwargs_proj["group"] =  om_data.proj_info.group

    da_arr = read_file(infile, is_tiff=om_data.raw_is_tiff, zoom_level=om_data.zoom_level, is_raw=True, resize_to_20x=False, sizes=om_data.sizes, **kwargs_proj)
    da_arr_ = crop(da_arr, np_coords=np_coords)

    if overwrite_cache:
        write_qptiff_2d(out_file, da_arr_.compute(), sizes=sizes)
    
    return da_arr_


def get_crop_coords(om_data: om.Omni3D, i_layer: int, overwrite_cache=False, zoom_level_crop_bg=-1, **kwargs_proj):
    dict_file_name = StageSampleTag.RAW.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info, check_exist=False)
    outfile = dict_file_name["raw"]
    out_dir = os.path.dirname(outfile)
    sample_name = om_data.proj_info.get_sample_name(i_layer)
    
    if os.path.isfile(f"{out_dir}/crop/{sample_name}_coords.npy") and not overwrite_cache:
        np_coords = np.load(f"{out_dir}/crop/{sample_name}_coords.npy")
        return np_coords

    
    dict_file_name = StageSampleTag.DATA.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info)
    infile = dict_file_name["data"]

    da_arr_ = read_file(infile, is_tiff=om_data.raw_is_tiff, zoom_level=zoom_level_crop_bg, is_raw=True, resize_to_20x=False, sizes=om_data.sizes, **kwargs_proj)
    kwargs_crop = kwargs_proj.get("calculate_crop_margin_dino", om_data.config["pp"]["crop"])
    is_HE = om_data.proj_info.get_dtype(i_layer=i_layer) == "HE"
    if not is_HE:
        da_arr_ = da.from_array(om.pl.imshow_IHC(da_arr_))
    # da_arr_ = rgb_he_to_gray(da_arr_, is_HE=is_HE, **om_data.config["datasets"].get("to_gray", {}))
    # da_arr_ = np.repeat(da_arr_, 3, 2)
    # kwargs_crop["cos_sim_cutoff"] = 0.6

    np_mask, np_coords, emb_sub = calculate_crop_margin_dino(da_arr_, **kwargs_crop)
    if overwrite_cache:
        np_mask_shown = np.repeat((255*np_mask).astype(np.uint8)[:, :, np.newaxis], 3, 2)
        
        i_beg, j_beg, i_end, j_end = np_coords
        h_, w_ = np_mask.shape
        i_beg = int(i_beg*h_)
        j_beg = int(j_beg*w_)
        i_end = int(i_end*h_)
        j_end = int(j_end*w_)
        np_mask_shown[:, :, 0] = 255*np_mask
        np_mask_shown[0:i_beg, :, 1] = 100
        np_mask_shown[i_end:, :, 1] = 100
        np_mask_shown[:, 0:j_beg, 1] = 100
        np_mask_shown[:, j_end:, 1] = 100
        fig = om.pl.plt.figure()
        ax = fig.add_subplot(1,2,1)
        ax.imshow(da_arr_)
        ax.set_title(f"{sample_name}")
        ax = fig.add_subplot(1,2,2)
        ax.imshow(np_mask_shown)
        ax.set_title(f"{om_data.group}-{np_coords}")
        
        fig.savefig(f"{out_dir}/crop/{om_data.group}-{sample_name}_mask.png")
        # cv2.imwrite(f"{out_dir}/crop/{sample_name}_mask.png", np_mask_shown)
        np.save(f"{out_dir}/crop/{sample_name}_coords.npy", np_coords)
        np.save(f"{out_dir}/crop/{sample_name}_embsub-{zoom_level_crop_bg}.npy", emb_sub)

    return np_coords