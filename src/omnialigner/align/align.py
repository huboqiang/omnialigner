from typing import Tuple

import dask.array as da
import torch
import numpy as np
from dask_image.ndinterp import affine_transform

import omnialigner as om
from omnialigner.preprocessing.pad import _modify_image_size
from omnialigner.logging import logger as logging
from omnialigner.utils.image_transform import apply_tfrs_to_dask
from omnialigner.utils.field_transform import warp_tensor, disp_to_grid, calculate_M_from_theta, tfrs_inv, resample_displacement_field_to_size, generate_grid
from omnialigner.utils.point_transform import raw_landmarks_to_padded, transform_keypoints, warp_landmark_grid

from omnialigner.dtypes import Dask_image_HWC, Np_kpts_N_yx_raw, Np_image_HWC, Tensor_tfrs, Tensor_trs, Np_disp_2d, Tensor_disp_2d


def apply_image_HD(
        image: Dask_image_HWC|Np_image_HWC,
        crop_size: Tuple[int, int, int, int]=None,
        pad_size: Tuple[int, int, int, int]|Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]=None,
        target_pad_size: Tuple[int, int]=None,
        affine_matrix: np.ndarray=None,
        tensor_tfrs_1: Tensor_tfrs|Tensor_trs=None,
        tensor_tfrs_2: Tensor_tfrs|Tensor_trs=None,
        l_disps: list[Np_disp_2d|Tensor_disp_2d]=None,
        is_uint8: bool=None,
        **kwargs
    ) -> Dask_image_HWC:
    """
    Apply omnialigner transformation to image in step:
    1. crop image
    2. pad image
    3. scaling affine_matrix [[ky, 0, 0], [0, kx, 0], [0, 0, 1]]
    4. apply stack transformation, [np.deg2rad(angle), 0, 0, sx, sy, fx, fy]
    5. apply affine transformation, [theta, tx, ty, sx, sy]
    6. apply nonrigid transformation, disp
    7. resize image to original size

    
    Args:
        image: Input image.
        crop_size: Crop size.
        pad_size: Pad size.
        affine_matrix: Affine matrix.
        tensor_tfrs_1: Tensor transformation 1 (om.align.stack).
        tensor_tfrs_2: Tensor transformation 2 (om.align.affine).
        target_pad_size: Target pad size.
        l_disps: Displacement field (om.align.nonrigid).
        is_uint8: Whether the input image is uint8.
        **kwargs: Keyword arguments. Including {"mode": "constant", "constant_values": 0, "tile_size": 2048}.
    
    Returns:
        Output image.
    """
    if pad_size is None:
        pad_size = ((0, 0), (0, 0), (0, 0))
    
    if len(pad_size) == 4:
        pad_size = ((pad_size[3], pad_size[2]), (pad_size[1], pad_size[0]), (0, 0))

    if affine_matrix is None:
        affine_matrix = np.eye(3)
    
    if tensor_tfrs_1 is None:
        tensor_tfrs_1 = torch.zeros(7).float()
        tensor_tfrs_1[5] = 1.
        tensor_tfrs_1[6] = 1.

    if tensor_tfrs_2 is None:
        tensor_tfrs_2 = torch.zeros(7).float()
        tensor_tfrs_2[5] = 1.
        tensor_tfrs_2[6] = 1.


    h, w, c = image.shape
    if crop_size is not None:
        h_beg, h_end = int(h*crop_size[0]), int(h*crop_size[2])
        w_beg, w_end = int(w*crop_size[1]), int(w*crop_size[3])
        image = image[h_beg:h_end, w_beg:w_end, :]
        h, w, _ = image.shape

    kwargs_pad = {}
    kwargs_pad["mode"] = kwargs.get("mode", "constant")
    kwargs_pad["constant_values"] = kwargs.get("constant_values", 0)

    tile_size = kwargs.get("tile_size", 2048)
    logging.info(f"Processing apply_affine_HD_custom, processing {image.shape} with {kwargs}")

    ky, kx = affine_matrix[0, 0], affine_matrix[1, 1]
    size_ = ( int(image.shape[0] / ky), int(image.shape[1] / kx))
    l_channels = []
    for i_channel in range(image.shape[2]):
        da_channel = affine_transform(image[:, :, i_channel:i_channel+1], affine_matrix)[:size_[0], :size_[1]]
        l_channels.append(da_channel)
    
    da_zoomed = da.concatenate(l_channels, axis=2)
    da_image_pad = da.pad(da_zoomed, pad_width=pad_size, **kwargs_pad)

    if target_pad_size is not None:
        da_image_pad = _modify_image_size(da_image_pad, target_pad_size)

    da_affine_hd_HWC = apply_tfrs_to_dask(da_image_pad, [tensor_tfrs_1, tensor_tfrs_2], tile_size=(tile_size, tile_size, 1), **kwargs_pad)

    if l_disps is None:
        return da_affine_hd_HWC

    is_uint8 = da_affine_hd_HWC.dtype == np.uint8 if is_uint8 is None else is_uint8
    tensor_nonrigid = om.tl.im2tensor(da_affine_hd_HWC, is_uint8=is_uint8)

    for disp in l_disps:
        if isinstance(disp, np.ndarray):
            disp = disp_to_grid(disp)
    
        kwargs_nonrigid = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}
        if not is_uint8:
            kwargs_nonrigid["mode"] = "nearest"

        tensor_nonrigid = warp_tensor(tensor_nonrigid, disp, **kwargs_nonrigid)
    
    np_nonrigid = om.tl.tensor2im(tensor_nonrigid, is_uint8=is_uint8)    
    da_nonrigid = da.from_array(np_nonrigid)
    return da_nonrigid


def apply_keypoint_HD(
        image: Dask_image_HWC|Np_image_HWC,
        keypoints: Np_kpts_N_yx_raw,
        crop_size: Tuple[int, int, int, int]=None,
        pad_size: Tuple[int, int, int, int]=None,
        affine_matrix: np.ndarray=None,
        tensor_tfrs_1: Tensor_tfrs|Tensor_trs=None,
        tensor_tfrs_2: Tensor_tfrs|Tensor_trs=None,
        l_disps: list[Np_disp_2d|Tensor_disp_2d]=None,
        zoom_level: float=40.0,
        max_size: int=1280
):
    if pad_size is None:
        pad_size = (0, 0, 0, 0)

    if affine_matrix is None:
        affine_matrix = np.eye(3)
    
    if tensor_tfrs_1 is None:
        tensor_tfrs_1 = torch.zeros(7).float()
        tensor_tfrs_1[5] = 1.
        tensor_tfrs_1[6] = 1.

    if tensor_tfrs_2 is None:
        tensor_tfrs_2 = torch.zeros(7).float()
        tensor_tfrs_2[5] = 1.
        tensor_tfrs_2[6] = 1.

    if crop_size is None:
        crop_size = (0, 0, 1, 1)

    # h, w, c = image.shape
    # h_beg, h_end = int(h*crop_size[0]), int(h*crop_size[2])
    # w_beg, w_end = int(w*crop_size[1]), int(w*crop_size[3])
    # # image = image[h_beg:h_end, w_beg:w_end, :]
    # h, w, _ = image.shape

    coord_x, coord_y = keypoints[:, 0], keypoints[:, 1]

    x_beg, y_beg, x_end, y_end = crop_size
    h_raw = image.shape[0] / (x_end - x_beg)
    w_raw = image.shape[1] / (y_end - y_beg)
    x_beg = int(x_beg * h_raw)
    y_beg = int(y_beg * w_raw)
    np_crop = np.array([y_beg, x_beg])

    np_cells = np.array([ [x, y] for x,y in zip(coord_x, coord_y) ]) / zoom_level
    keypoints_raw = torch.from_numpy(np_cells - np_crop).float()
    kpt_zoom_level_smallest = keypoints_raw #(om_data.l_scales[zoom_level] / om_data.l_scales[-1])
    pad_ratio = torch.FloatTensor([1/affine_matrix[0, 0], 1/affine_matrix[1, 1]])
    lm_pad = raw_landmarks_to_padded(kpt_zoom_level_smallest, ratio=pad_ratio, padded_size=pad_size)
    grid_M_1 = calculate_M_from_theta(tfrs_inv(tensor_tfrs_1), h=max_size, w=max_size)[0:2]
    grid_M_2 = calculate_M_from_theta(tfrs_inv(tensor_tfrs_2), h=max_size, w=max_size)[0:2]
    lm_M_step1 = transform_keypoints(lm_pad, grid_M_1)
    kpt_affine = transform_keypoints(lm_M_step1, grid_M_2)
    if l_disps is None:
        return kpt_affine.cpu().numpy()

    for disp in l_disps:
        if isinstance(disp, np.ndarray):
            disp = disp_to_grid(disp)
    
        kwargs_nonrigid = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}
        if not is_uint8:
            kwargs_nonrigid["mode"] = "nearest"
            
        with torch.no_grad():
            displacement_field_smoothed_ = resample_displacement_field_to_size(disp, (max_size, max_size))
            grid = generate_grid(tensor=displacement_field_smoothed_)
            kpts_new_scaled = warp_landmark_grid(kpt_affine / max_size, grid=grid+displacement_field_smoothed_)
            out_kpt = kpts_new_scaled[:, 0, :]
            out_kpt = out_kpt * max_size
    
    return out_kpt.detach().cpu().numpy()