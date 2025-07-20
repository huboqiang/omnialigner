
import os
import multiprocessing as mp

import torch
from typing import Tuple
import dask.array as da
import torch.nn.functional as F
import numpy as np
import scanpy as sc

import omnialigner as om
from omnialigner.dtypes import Np_kpts_N_yx_raw, Dask_image_HWC, Tuple_bbox
from omnialigner.cache_files import StageTag, StageSampleTag
from omnialigner.logging import logger as logging
from omnialigner.utils.tiles import merge_from_overlapped_tiles
from omnialigner.utils.field_transform import grid_2d_to_disp_field, disp_field_to_grid_2d
from omnialigner.utils.grid_sample import mp_array_grid_sample
from omnialigner.keypoints.keypoints import init_default_detector, KeypointDetectorMeta
from omnialigner.keypoints import KeypointPairs


def generate_tiles(
        np_kpts: Np_kpts_N_yx_raw,
        tile_size: int=2048,
        extend_ratio: float=0.1,
        threshold_tile: int=10,
        max_size: Tuple[int, int] = (16000, 16000)
    ):
    extend_len = int(tile_size * extend_ratio / 2)
    l_tiles = []
    l_tiles_extend = []
    l_masks = []
    
    kpt_up_left = np_kpts.min(axis=0).astype(int)
    kpt_down_right = np_kpts.max(axis=0).astype(int)
    for y_beg in range(kpt_up_left[1], kpt_down_right[1], tile_size):
        for x_beg in range(kpt_up_left[0], kpt_down_right[0], tile_size):
            x_beg = max(x_beg, 0)
            y_beg = max(y_beg, 0)
            x_end = min(x_beg + tile_size, max_size[0])
            y_end = min(y_beg + tile_size, max_size[1])
            np_selected = np_kpts[
                (np_kpts[:, 1] >= y_beg) & (np_kpts[:, 1] < y_end) &
                (np_kpts[:, 0] >= x_beg) & (np_kpts[:, 0] < x_end)
            ]
            n_selected = np_selected.shape[0]
            if n_selected < threshold_tile:
                continue
            
            x_beg_extend = max(x_beg - extend_len, 0)
            y_beg_extend = max(y_beg - extend_len, 0)
            x_end_extend = min(x_beg + tile_size + extend_len, max_size[0])
            y_end_extend = min(y_beg + tile_size + extend_len, max_size[1])
            l_tiles.append([x_beg, y_beg, x_end, y_end])
            l_tiles_extend.append([x_beg_extend, y_beg_extend, x_end_extend, y_end_extend])
            l_masks.append(n_selected)

    return l_tiles, l_tiles_extend, l_masks


def refine_tile(
        da_ref: Dask_image_HWC,
        da_query: Dask_image_HWC,
        crop_region: Tuple_bbox,
        out_ovlp_file: str = None,
        detector: KeypointDetectorMeta = None
    ) -> Tuple[torch.Tensor, torch.Tensor, KeypointPairs]:
    """
    Refine the tile by applying non-rigid transformation and matching keypoints.
    
    Args:
        da_ref (Dask_image_HWC): Reference image data.
        da_query (Dask_image_HWC): Query image data.
        crop_region (Tuple_bbox): The region to crop in the format (x_beg, y_beg, x_end, y_end).
    
    Returns:
        tensor_moved_roi: The transformed query image tensor.
        disp: The displacement field tensor.
        kd: Keypoint pairs containing matched keypoints and images.
    """
    if detector is None:
        detector = init_default_detector()

    x_beg, y_beg, x_end, y_end = crop_region
    h, w = y_end - y_beg, x_end - x_beg
    da_ref_roi = da_ref[y_beg:y_end, x_beg:x_end]
    da_query_roi = da_query[y_beg:y_end, x_beg:x_end]
    if da_ref_roi.shape[-1] == 1:
        da_ref_roi = np.repeat(da_ref_roi, 3, 2)

    if da_query_roi.shape[-1] == 1:
        da_query_roi = np.repeat(da_query_roi, 3, 2)

    tensor_ref_roi = om.tl.im2tensor(da_ref_roi)
    tensor_query_roi = om.tl.im2tensor(da_query_roi)

    mkpt_F, mkpt_M, _,roma_model = om.kp.match(tensor_ref_roi, tensor_query_roi, detector=detector)
    kd = om.kp.KeypointPairs(
        image_F=tensor_ref_roi,
        image_M=tensor_query_roi,
        mkpts_F=mkpt_F,
        mkpts_M=mkpt_M,
        index_matches=range(len(mkpt_F))
    )

    grid_2d = F.interpolate(roma_model[1]["flow"][0:1, :, :, :], size=(tensor_query_roi.shape[2], tensor_query_roi.shape[3]), mode="bilinear", align_corners=True)
    grid_2d = grid_2d.permute(0, 2, 3, 1)
    tensor_moved_roi = F.grid_sample(tensor_query_roi, grid_2d, "bilinear", align_corners=True)
    disp = grid_2d_to_disp_field(grid_2d)

    if out_ovlp_file is not None:
        import cv2
        img_shown = 3*da_ref_roi.compute()
        img_shown[:, :, 2] = om.tl.tensor2im(tensor_moved_roi)[..., 0]
        img_shown[:, :, 1] = 0
        cv2.imwrite(out_ovlp_file, img_shown)

    return tensor_moved_roi, disp, kd

def show_bbox_on_image(image, bbox_list, labels=None, colors=None, figsize=(12, 8)):
    """
    Display bounding boxes on an image.

    Args:
        image: numpy array image (H, W) or (H, W, C)
        bbox_list: list of tuples, each as (x_beg, y_beg, x_end, y_end)
        labels: list of str, optional labels for each box
        colors: list of str, optional colors for each box
        figsize: tuple, figure size
    """
    import matplotlib.patches as patches
    plt = om.pl.plt
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

    while len(colors) < len(bbox_list):
        colors.extend(colors)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    for i, bbox in enumerate(bbox_list):
        x_beg, y_beg, x_end, y_end = bbox
        width = x_end - x_beg
        height = y_end - y_beg
        rect = patches.Rectangle(
            (x_beg, y_beg), width, height,
            linewidth=2, edgecolor=colors[i % len(colors)],
            facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        label = labels[i] if labels and i < len(labels) else f'Box {i}'
        ax.text(
            x_beg, y_beg - 10, label,
            color=colors[i % len(colors)], fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
        )

    ax.set_title(f'Image with {len(bbox_list)} Bounding Boxes')
    plt.tight_layout()
    plt.show()



def nonrigid_tiles(om_data: om.Omni3D, overwrite_cache=False):
    """
    Perform non-rigid alignment on image tiles and merge the results.

    Args:
        om_data (om.Omni3D): 
            The Omni3D object containing project configuration and data.
        overwrite_cache (bool, optional): 
            Whether to overwrite cached results. Defaults to False.

    Returns:
        None: 
            Saves the merged displacement field to a file.
    """
    dict_file_disp = StageTag.NONRIGID_TILES.get_file_name(projInfo=om_data.proj_info, check_exist=False)

    dict_align = om_data.config["align"]
    config_info = dict_align.get("nonrigid_tiles", dict_align["nonrigid"])
    tile_size = config_info.get("tile_size", 2048)
    extend_ratio = config_info.get("extend_ratio", 0.1)
    zoom_level = config_info.get("zoom_level", 1)
    threshold_tile = config_info.get("threshold_tile", 30)
    root_dir = om_data.proj_info.root_dir
    proj_name = om_data.proj_info.project
    group_name = om_data.proj_info.group
    version = om_data.proj_info.version
    om_data.set_zoom_level(zoom_level)
    da_ref = om.align.apply_nonrigid_HD(om_data, i_layer=0, tag="gray", overwrite_cache=False)
    file_h5ad = f"{root_dir}/analysis/{proj_name}/{version}/11.single_cell/{group_name}/single_cell_exp.h5ad"
    adata = sc.read_h5ad(file_h5ad)
    
    max_size = (om_data.max_size * om_data.l_scales[zoom_level], 
                om_data.max_size * om_data.l_scales[zoom_level])
    if not os.path.exists(f'{dict_file_disp["out_dir"]}/tiles.pth') or overwrite_cache:
        l_tiles, l_tiles_ext, l_masks = generate_tiles(
                    adata.obsm["spatial"],
                    tile_size=tile_size,
                    extend_ratio=extend_ratio,
                    threshold_tile=threshold_tile,
                    max_size=max_size
        )
        file_tiles = f"{dict_file_disp['out_dir']}/tiles.pth"
        dict_tiles = {
            "tiles": l_tiles,
            "tiles_ext": l_tiles_ext,
            "l_masks": l_masks,
        }
        torch.save(dict_tiles, file_tiles, _use_new_zipfile_serialization=False)

    dict_tiles = torch.load(f"{dict_file_disp['out_dir']}/tiles.pth")
    l_tiles_ext = dict_tiles["tiles_ext"]

    for i_layer in range(1, len(om_data)):
        if os.path.exists(f'{dict_file_disp["out_dir"]}/disp_{i_layer}_merged.pth') and not overwrite_cache:
            continue

        da_query = om.align.apply_nonrigid_HD(om_data, i_layer=i_layer, tag="rmaf", overwrite_cache=False)[:, :, 0:1]
        l_disp = []
        for crop_region in l_tiles_ext:
            out_roi_prefix = f'{dict_file_disp["out_dir"]}/roi_{crop_region[0]}_{crop_region[1]}'
            if not os.path.exists(f"{out_roi_prefix}/disp_{i_layer}_roi.pth"):
                os.makedirs(out_roi_prefix, exist_ok=True)
                tensor_moved_roi, disp, kd = refine_tile(da_ref, da_query, crop_region, out_ovlp_file=f"{out_roi_prefix}/ovlp_{i_layer}.png")


                torch.save(disp, f"{out_roi_prefix}/disp_{i_layer}_roi.pth")

            disp = torch.load(f"{out_roi_prefix}/disp_{i_layer}_roi.pth")
            l_disp.append(disp)

        size = om_data.max_size * om_data.l_scales[zoom_level]
        disp_merged = merge_from_overlapped_tiles(l_tiles_ext, l_disp=l_disp, H=size, W=size)
        torch.save(disp_merged, f'{dict_file_disp["out_dir"]}/disp_{i_layer}_merged.pth', _use_new_zipfile_serialization=False)


def apply_nonrigid_tiles_HD(
        om_data: om.Omni3D,
        i_layer:int=0,
        tag:str="rmaf",
        overwrite_cache=False,
        **kwargs
    ) -> Dask_image_HWC:
    """Apply non-rigid (TILE LEVEL) transformation to high-definition images.

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
        **kwargs: Additional arguments passed to transformation functions.

    Returns:
        Dask_image_HWC: Transformed high-definition image.

    Note:
        - Supports different data types with appropriate processing
        - Uses dask arrays for memory-efficient processing
        - Results are cached based on data type and zoom level
        - Handles both uint8 and float data types
    """
    file_nonrigid_tiles = StageSampleTag.NONRIGID_TILES_HD.get_file_name(
            i_layer=i_layer, projInfo=om_data.proj_info, check_exist=False
    )["zarr"]

    if os.path.exists(file_nonrigid_tiles) and not overwrite_cache:
        logging.info(f"Loading nonrigid_tiles transformation of raw image from {file_nonrigid_tiles}")
        da_nonrigid = da.from_zarr(file_nonrigid_tiles)
        return da_nonrigid

    dict_nonrigid_tiles = StageTag.NONRIGID_TILES.get_file_name(
        projInfo=om_data.proj_info,
        check_exist=False
    )
    if dict_nonrigid_tiles is None:
        err_msg = "No nonrigid tiles found in the project configuration.\n"
        err_msg += "Running om.align.nonrigid_tiles() to generate nonrigid tiles first."
        raise ValueError(err_msg)

    da_nonrigid = om.align.apply_nonrigid_HD(
        om_data, i_layer=i_layer, tag=tag, overwrite_cache=overwrite_cache
    )
    if not os.path.exists(f'{dict_nonrigid_tiles["out_dir"]}/disp_{i_layer}_merged.pth'):
        return da_nonrigid

    n_cpus = mp.cpu_count()
    input_tensor = om.tl.im2tensor(da_nonrigid)
    disp_merged = torch.load(
        f'{dict_nonrigid_tiles["out_dir"]}/disp_{i_layer}_merged.pth'
    )
    grid = disp_field_to_grid_2d(disp_merged.float())
    result = mp_array_grid_sample(input_tensor, grid, chunks=(1, 'auto', 1024, 1024), max_workers=n_cpus)
    
    if isinstance(result, np.ndarray):
        result = da.from_array(result, chunks=(1, 'auto', 1024, 1024))

    da_nonrigid_tiles = (255*np.moveaxis(result[0], 0, -1)).astype(np.uint8)
    da_nonrigid_tiles.to_zarr(file_nonrigid_tiles, overwrite=True)
    return da_nonrigid_tiles
