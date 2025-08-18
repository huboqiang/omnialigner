from typing import Tuple

import dask.array as da
import torch
import numpy as np
from dask_image.ndinterp import affine_transform

import omnialigner as om
from omnialigner.preprocessing.pad import _modify_image_size
from omnialigner.logging import logger as logging
from omnialigner.utils.image_transform import apply_tfrs_to_dask
from omnialigner.utils.field_transform import warp_tensor, disp_to_grid
from omnialigner.dtypes import Dask_image_HWC, Np_image_HWC, Tensor_tfrs, Tensor_trs, Np_disp_2d, Tensor_disp_2d


def apply_image_HD(
        image: Dask_image_HWC|Np_image_HWC,
        crop_size: Tuple[int, int, int, int]=None,
        pad_size: Tuple[int, int, int, int]=None,
        target_pad_size: Tuple[int, int]=None,
        affine_matrix: np.ndarray=None,
        tensor_tfrs_1: Tensor_tfrs|Tensor_trs=None,
        tensor_tfrs_2: Tensor_tfrs|Tensor_trs=None,
        l_disps: list[Np_disp_2d|Tensor_disp_2d]=None,
        **kwargs
    ) -> Dask_image_HWC:
    f"""
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
        disp: Displacement field (om.align.nonrigid).
        **kwargs: Keyword arguments. Including {"mode", "constant", "constant_values", 0, "tile_size", 2048}.
    
    Returns:
        Output image.
    """
    if pad_size is None:
        pad_size = ((0, 0), (0, 0), (0, 0))

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

    if crop_size is not None:
        h, w, c = image.shape
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
    size_ = ( int(image.shape[0] // ky), int(image.shape[1] // kx))
    da_zoomed = affine_transform(image, affine_matrix)[:size_[0], :size_[1], :]
    da_image_pad = da.pad(da_zoomed, pad_width=pad_size, **kwargs_pad)

    if target_pad_size is not None:
        da_image_pad = _modify_image_size(da_image_pad, target_pad_size)

    da_affine_hd_HWC = apply_tfrs_to_dask(da_image_pad, [tensor_tfrs_1, tensor_tfrs_2], tile_size=(tile_size, tile_size, 1), **kwargs_pad)

    if l_disps is None:
        return da_affine_hd_HWC

    is_uint8 = da_affine_hd_HWC.dtype == np.uint8
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