import json

import cv2
from tqdm import tqdm
import numpy as np
import dask.array as da
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from omnialigner.omni_3D import Omni3D
from omnialigner.align.nonrigid_HD import apply_nonrigid_HD
from omnialigner.plotting.matplotlib_init import plt


def init_color_map():
    wavelengths = {
        'DAPI': (0, 0, 1),
        'Opal 480': (0, 1, 1),
        'Opal 520': (0, 1, 0),
        'Opal 570': (1, 1, 0),
        'Opal 620': (1, 0.5, 0),
        'Opal 690': (1, 0, 0),
        'Opal 780': (1, 1, 1)
    }

    colormaps = {"Sample AF": "gray"}
    for name, color in wavelengths.items():
        cmap = LinearSegmentedColormap.from_list(name, ['black', color])
        colormaps[name] = cmap

    return colormaps


def init_color_list():
    import os
    dirname = os.path.dirname(os.path.abspath(__file__))
    file_json = os.path.join(dirname, "../configs/IHC_layer_info.json")
    with open(file_json, "r") as f:
        dict_IHC_layer = json.load(f)

    colormaps = init_color_map()
    colorList_IHC = [colormaps[laser_nm]
                     for laser_nm in dict_IHC_layer["order_in_tiff"][:-1]]
    return colorList_IHC


def apply_colormap(data, cmap, percent=95, alpha_enhance=1.0):
    # max_val = np.percentile(data[data > 0], percent)
    max_val = np.percentile(data[data > 0], percent) if np.any(data > 0) else 0
    norm_data = alpha_enhance * (data - data.min()) / (max_val - data.min())
    rgba = cmap(norm_data)
    rgb = rgba[:, :, :3]
    return rgb


def apply_colormap_dask(da_image, colorList=None, percent=95, alpha_enhance=1.0):
    """Apply colormap to IHC (Immunohistochemistry) images using dask.

    Args:
        da_image: Input dask array image.
        colorList: Optional list of colormaps for each channel.
        percent: Percentile for normalization (default: 95).
        alpha_enhance: Alpha enhancement factor (default: 1.0).

    Returns:
        numpy.ndarray: RGB image with applied colormap.
    """
    if da_image.shape[2] == 3:
        return da_image
    

    if colorList is None:
        colorList = init_color_list()
    
    combined_rgb = np.zeros(
        (da_image.shape[0], da_image.shape[1], 3), dtype=np.float32)
    for i_layer in range(da_image.shape[2]-1):
        rgb = apply_colormap(
            da_image[:, :, i_layer], colorList[i_layer], percent=percent, alpha_enhance=alpha_enhance)
        combined_rgb += rgb

    combined_rgb = combined_rgb / combined_rgb.max()
    combined_rgb = np.clip(combined_rgb, 0, 1)
    combined_rgb_uint8 = (combined_rgb * 255).astype(np.uint8)
    return combined_rgb_uint8


def extract_sub_tags(om_data):
    dict_sub_tags = {}
    df_sample = om_data.proj_info.df_proj_info
    for tag in df_sample["type"].unique():
        dict_sub_tags[tag] = df_sample[df_sample["type"]
                                       == tag]["group_idx"].values

    return dict_sub_tags


def make_dask_3d_sub(om_data: Omni3D, dtype: str = "P1", region_xyz: list = [6000, 8000, 4000, 6000, 0, -1]):
    dict_sub_tags = extract_sub_tags(om_data)

    ratio = 1
    if dtype == "HE":
        ratio = 4

    l_dask_rmaf = []
    for i_layer in tqdm(dict_sub_tags[dtype]):
        tag = "raw" if dtype == "HE" else "rmaf"
        zoom_level = 0 if dtype == "HE" else 1
        om_data.set_zoom_level(zoom_level)
        da_nonrigid = apply_nonrigid_HD(
            om_data, i_layer=i_layer, tag=tag, overwrite_cache=False)
        l_dask_rmaf.append(da_nonrigid)

    da_nonrigid = da.stack(l_dask_rmaf, axis=0)
    da_3d_sub = da_nonrigid[:, ratio*region_xyz[0]:ratio *
                            region_xyz[1], ratio*region_xyz[2]:ratio*region_xyz[3], :]
    return da_3d_sub


def normalize_layer(np_img, i_ref=0, axis_norm=1, blur_size=None):
    """
    axis_norm: 0 for layer, 1 for channel
    Args:
        np_img: [Z, H, W, C] 2D image array
        axis_norm: 1 for H and 2 for W
    """
    if axis_norm == 1:
        np_img_sub = np_img[:, 0, :, :]
    elif axis_norm == 2:
        np_img_sub = np_img[:, :, 0, :]
    elif axis_norm == -1:
        np_img_sub = np_img[:, -1, :, :]
    elif axis_norm == -2:
        np_img_sub = np_img[:, :, -1, :]

    else:
        raise ValueError(f"axis_norm must be +-1 or +-2, but got {axis_norm}")
    layer_mean = np_img_sub.mean(1)
    layer_std = np_img_sub.std(1)
    np_3d_norm = np_img_sub - layer_mean[:,  None, :]
    np_3d_norm = np_3d_norm / layer_std[:, None, :]
    np_3d_norm = np_3d_norm * \
        layer_std[i_ref:i_ref+1, None, :] + layer_mean[i_ref:i_ref+1, None, :]
    np_3d_norm = np.clip(np_3d_norm.compute(), 0, 255).astype(np.uint8)
    if blur_size is not None:
        np_3d_norm = cv2.blur(np_3d_norm, blur_size)
    return np_3d_norm


def plot_standard_3d_5view(da_3d_sub, tag: str = "HE", colorList: list = None, blur_size=None):
    if colorList is None:
        colorList = init_color_list()

    fig = plt.figure(figsize=(15, 15))
    ax2 = fig.add_subplot(3, 3, 2)  # u
    ax4 = fig.add_subplot(3, 3, 4)  # l
    ax5 = fig.add_subplot(3, 3, 5)  # f
    ax6 = fig.add_subplot(3, 3, 6)  # r
    ax8 = fig.add_subplot(3, 3, 8)  # d
    

    img_up = normalize_layer(
        da_3d_sub, i_ref=0, axis_norm=1, blur_size=blur_size)
    img_left = normalize_layer(
        da_3d_sub, i_ref=0, axis_norm=2, blur_size=blur_size)
    img_front = da_3d_sub[0, :, :, :]
    img_right = normalize_layer(
        da_3d_sub, i_ref=0, axis_norm=-2, blur_size=blur_size)
    img_down = normalize_layer(
        da_3d_sub, i_ref=0, axis_norm=-1, blur_size=blur_size)
    if tag != "HE":
        img_up = apply_colormap_dask(img_up, colorList)
        img_left = apply_colormap_dask(img_left, colorList)
        img_front = apply_colormap_dask(img_front, colorList)
        img_right = apply_colormap_dask(img_right, colorList)
        img_down = apply_colormap_dask(img_down, colorList)

    ax2.imshow(img_up, aspect="auto")
    ax2.invert_yaxis()

    ax4.imshow(np.moveaxis(img_left, 0, 1), aspect="auto")
    ax4.invert_xaxis()

    ax5.imshow(img_front, aspect="auto")

    ax6.imshow(np.moveaxis(img_right, 0, 1), aspect="auto")

    ax8.imshow(img_down, aspect="auto")

    plt.close()
    return fig
