import os
from typing import List

import pandas as pd
import dask.array as da
import numpy as np
import cv2

from omnialigner.logging import logger as logging
from omnialigner.utils.io import read_ome_tiff
from omnialigner.utils.image_transform import resize_dask

dict_projects = {"panlab": "i_page", "ANHIR": "i_page", "CRC": "i_level", "Acrobat": "i_level"}
dict_max_size = {
    "ANHIR": {"default": 1600},
    "Acrobat": {"default": 1600},
    "panlab": {"default": 2500, "pdac": 1600},
    "CRC": {"default": 2000},
}

dict_subplot_num = {
    "ANHIR": {"default": (30, 30)},
    "Acrobat": {"default": (30, 30)},
    "panlab": {"default": (10, 10), "pdac": (30, 30)},
    "CRC": {"default": (7, 7)},
}


def read_file(
        file_name: str,
        zoom_level: int=0,
        is_raw: bool=True,
        resize_to_20x: bool=False,
        is_tiff: bool=False,
        sizes: List[int]=[1, 4, 8, 40],
        **kwargs) -> da.Array:
    """
    Read input file and store it into FILE_cache["RAW"]

    - **`file_name` will be read using `cv2.imread()` by default,
    - if you want to read ome.tiff, these functions will be checked in order:
        - set a existing reading function in "project" and "group" in `kwargs["datasets"]`
        - specify `read_ome_tiff` in `kwargs["datasets"]`, e.g. `read_ome_tiff: {"i_page": 0, "i_level": 6, "l_channels": [0,1,2]}`

    Args:
        file_name: str, path to the input file
        zoom_level: int, zoom level for the image
        is_raw: bool, whether to return raw data
        resize_to_20x: bool, whether to resize to 20x magnification
        is_tiff: bool, whether the file is tiff
        sizes: List[int], sizes for the image
        kwargs: dict, additional arguments for read_ome_tiff
            
    Returns:
        da.Array, da.array of the image
    """
    if not is_tiff:
        logging.info(f"Reading {file_name} using cv2.imread()")
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        zoom_level = kwargs.get("zoom_level", 0)
        if zoom_level == 0:
            return da.from_array(img)

        resized_wh = (img.shape[1]//sizes[zoom_level], img.shape[0]//sizes[zoom_level])
        img = cv2.resize(img, resized_wh)
        return da.from_array(img)

    project = kwargs.get("project", "")
    group = kwargs.get("group", "")
    func = generate_WSI(project, group)
    if func is not None:
        logging.info(f"Reading {file_name} using {func.__name__}")
        da_arr = func(file_name, zoom_level=zoom_level, resize_to_20x=resize_to_20x, is_raw=is_raw)
        return da_arr

    kwargs_read_ome_tiff = kwargs.get("read_ome_tiff", {})
    logging.info(f"Reading {file_name} using read_ome_tiff, {kwargs_read_ome_tiff}")
    da_arr = read_ome_tiff(file_name, **kwargs_read_ome_tiff)
    return da_arr

def read_scaled_landmark(file_name):
    if os.path.isfile(file_name):
        return pd.read_csv(file_name, index_col=[0]).values

    return None

def load_datainfo_df(file_data):
    df_project = pd.read_csv(file_data)
    return df_project




class WSIGeneratorFactory:
    @staticmethod
    def create_generator(project_name="default", group_name=""):
        """Factory method to create appropriate WSI generator"""
        generator_name = f'generate_{project_name}'
        generator_name_group = f'generate_{project_name}_{group_name}'
        
        if hasattr(WSIGenerator, generator_name_group):
            return getattr(WSIGenerator, generator_name_group)
        elif hasattr(WSIGenerator, generator_name):
            return getattr(WSIGenerator, generator_name)
        
        logging.warning(f"No such generator method in WSIGenerator: {generator_name}")
        return None

class WSIGenerator:
    """
    A class for generating WSI (Whole Slide Image) data with standardized parameters.
    All generator methods must implement these parameters:
        tiff (str): Path to the tiff file
        zoom_level (int): Zoom level for the image
        (Optional) is_raw (bool): Whether to return raw data
        (Optional) resize_to_20x (bool): Whether to resize to 20x magnification
    """
    
    @classmethod
    def __init_subclass__(cls):
        """Enforce parameter requirements on all generator methods"""
        for name, method in cls.__dict__.items():
            if name.startswith('generate_') and callable(method):
                # Check if method has required parameters
                import inspect
                params = inspect.signature(method).parameters
                required = {"tiff", "zoom_level"}
                if not required.issubset(params):
                    raise TypeError(
                        f"Generator method {name} must include parameters: {required}"
                    )
    @staticmethod
    def generate_fair(file_name, zoom_level=0, resize_to_20x=True, is_raw=False, **kwargs):
        return read_ome_tiff(file_name, i_page=zoom_level, i_level=0, l_channels=[0], axes=[0, 1, 3])
    

    @staticmethod
    def generate_panlab(tiff, zoom_level=0, resize_to_20x=True, is_raw=False, **kwargs):
        kwargs = {
            "i_level": 0,
            "i_page": zoom_level
        }
        if tiff.split(".")[-1] == "qptiff":
            kwargs = {
                "i_level": zoom_level,
                "i_page": 0
            }

        try:
            l_channels = list(range(8)) if is_raw else [7]
            da_arr = read_ome_tiff(tiff, l_channels=l_channels, **kwargs)
            if not is_raw:
                da_arr = da.repeat(da_arr, 3, axis=2)
        except:
            da_arr = read_ome_tiff(tiff, l_channels=[0, 1, 2], **kwargs)

        #     except:
        #         l_channels = list(range(8)) if is_raw else [7]
        #         da_arr = read_ome_tiff(tiff, l_channels=l_channels, **kwargs)
        #         if not is_raw:
        #             da_arr = da.repeat(da_arr, 3, axis=2)
        if not resize_to_20x:
            return da_arr
        
        wsi = cv2.resize(da_arr.compute(), (da_arr.shape[1]//2, da_arr.shape[0]//2))
        return da.from_array(wsi)

    @staticmethod
    def generate_panlab_ipmn(tiff, zoom_level=0, resize_to_20x=True, is_raw=False, **kwargs):
        try:
            l_channels = list(range(8)) if is_raw else [7]
            da_arr = read_ome_tiff(tiff, i_level=0, i_page=zoom_level, l_channels=l_channels)
            if is_raw:
                da_arr = 3*da.repeat(da_arr, 3, axis=2)
        except:
            da_arr = read_ome_tiff(tiff, i_level=0, i_page=zoom_level, l_channels=[0, 1, 2])

        if not resize_to_20x:
            return da_arr

        wsi = cv2.resize(da_arr.compute(), (da_arr.shape[1]//2, da_arr.shape[0]//2))
        return da.from_array(wsi)

    @staticmethod
    def _generate_CRC_HE(tiff, zoom_level=1, is_raw=False, **kwargs):
        try:
            da_arr = read_ome_tiff(tiff, i_level=zoom_level)
            return da_arr
        except:
            return None

    @staticmethod
    def _generate_CRC_Cycif(tiff, zoom_level=0, is_raw=False, **kwargs):
        l_channels = list(range(40)) if is_raw else [1] 
        da_arr = read_ome_tiff(tiff, i_level=zoom_level, l_channels=l_channels)
        da_arr_small = read_ome_tiff(tiff, i_level=-1, l_channels=l_channels)
        max_uint16 = da.percentile(da_arr_small.ravel(), 99).compute()
        np_arr = (255*(da_arr/max_uint16)).astype(np.uint8)
        wsi = np_arr if is_raw else da.repeat(np_arr, 3, axis=2)
        return wsi

    @staticmethod
    def generate_CRC(tiff, zoom_level=0, is_raw=False, **kwargs):
        da_arr = WSIGenerator._generate_CRC_HE(tiff, zoom_level=zoom_level, is_raw=is_raw)
        if da_arr is None or da_arr.dtype == np.uint16:
            zoom_level_ = max(0, zoom_level-1)
            da_arr = WSIGenerator._generate_CRC_Cycif(tiff, zoom_level=zoom_level_, is_raw=is_raw)
            if zoom_level == 0:
                da_arr = resize_dask(da_arr, zoom=2)
        return da_arr

    @staticmethod
    def generate_ANHIR(tiff, zoom_level=0, is_raw=False, **kwargs):
        da_arr = read_ome_tiff(tiff, i_level=0, i_page=zoom_level)
        return da_arr

    @staticmethod
    def generate_Acrobat(tiff, zoom_level=0, is_raw=False, **kwargs):
        da_arr = read_ome_tiff(tiff, i_level=zoom_level, i_page=0, l_channels=[0])
        return da_arr[:,:,0,:]

# For backwards compatibility
generate_WSI = WSIGeneratorFactory.create_generator
