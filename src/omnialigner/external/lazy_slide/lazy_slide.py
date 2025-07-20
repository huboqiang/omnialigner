import os

import spatialdata as sd
import lazyslide as zs
from wsidata import open_wsi

import omnialigner as om
from omnialigner.cache_files import StageSampleTag

def run_lazy_slide(om_data: om.Omni3D, i_layer: int=0, overwrite_cache=False) -> sd.SpatialData:
    dict_file_ZS = StageSampleTag.ZS_DATA.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)
    if not overwrite_cache and os.path.isdir(dict_file_ZS["zarr"]):
        sd_sub = sd.read_zarr(dict_file_ZS["zarr"])
        return sd_sub


    dict_file_RAW = StageSampleTag.DATA.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)
    slide = dict_file_RAW["data"]

    wsi = open_wsi(slide, store=None)
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, 64)
    zs.seg.cells(wsi)
    zs.seg.nulite(wsi)
    zs.pp.tile_tissues(wsi, 512, mpp=0.5, key_added="text_tiles")
    zs.tl.feature_extraction(wsi, "plip", tile_key="text_tiles")
    # if overwrite_cache and os.path.isdir(dict_file_ZS["zarr"]):
    import shutil
    if os.path.isdir(dict_file_ZS["zarr"]):
        shutil.rmtree(dict_file_ZS["zarr"])

    wsi.write(dict_file_ZS["zarr"], overwrite=overwrite_cache)
    
    return sd.read_zarr(dict_file_ZS["zarr"])
