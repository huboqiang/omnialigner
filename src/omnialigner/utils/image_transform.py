from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import dask_image.ndinterp
import dask.array as da

from omnialigner.utils.field_transform import tfrs_to_grid_M, calculate_M_from_theta
from omnialigner.dtypes import Tensor_image_NCHW, Dask_image_HWC

transform = transforms.Compose([
    transforms.ToTensor()
])

def transform_dask(da_image):
    tensor = transform(da_image.compute()).unsqueeze(0)
    return tensor


def apply_theta_grid(image_tensor: Tensor_image_NCHW, grid_M, tensor_size=None) -> Tensor_image_NCHW:
    if tensor_size is None:
        tensor_size = image_tensor.size()
    
    grid = torch.nn.functional.affine_grid(grid_M.unsqueeze(0), tensor_size, align_corners=True)
    return torch.nn.functional.grid_sample(image_tensor, grid, align_corners=True)


    
def create_pyramid(tensor: Tensor_image_NCHW, num_levels: int, mode: str='bilinear') -> Iterable[Tensor_image_NCHW]:
    """
    Creates the resolution pyramid of the input tensor (assuming uniform resampling step = 2).

    Parameters
    ----------
    tensor : Tensor_image_NCHW
        The input tensor
    num_levels: int
        The number of output levels
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    
    Returns
    ----------
    pyramid: list of Tensor_image_NCHW
        The created resolution pyramid

    """
    pyramid = [None]*num_levels
    for i in range(num_levels - 1, -1, -1):
        print("create_pyramid", i)
        if i == num_levels - 1:
            pyramid[i] = tensor
        else:
            current_size = pyramid[i+1].size()
            new_size = (int(current_size[j] / 2) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = torch.Size(new_size)[2:]
            new_tensor = resample_tensor_to_size(gaussian_smoothing(pyramid[i+1], 1), new_size, mode=mode)
            pyramid[i] = new_tensor
    return pyramid


def resample_tensor_to_size(tensor: Tensor_image_NCHW, new_size: torch.Tensor, mode: str='bilinear') -> Tensor_image_NCHW:
    """
    TODO
    """
    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)


def gaussian_smoothing(tensor : Tensor_image_NCHW, sigma : float) -> Tensor_image_NCHW:
    """
    TODO
    """
    with torch.set_grad_enabled(False):
        kernel_size = int(sigma * 2.54) + 1 if int(sigma * 2.54) % 2 == 0 else int(sigma * 2.54)
        return transforms.GaussianBlur(kernel_size, sigma)(tensor)


def apply_tfrs_to_dask(da_image: Dask_image_HWC, tensor_tfrs: List[torch.Tensor], tile_size=(2048, 2048), **kwargs_pad) -> Dask_image_HWC:
    """
    Convert a list of tensor_tfrs to a grid_M matrix.
    """
    M_3x3 = np.eye(3)
    for tfrs in tensor_tfrs:
        reordered_tfrs = torch.FloatTensor([-tfrs[0], tfrs[2], tfrs[1], tfrs[4], tfrs[3]])
        if len(tfrs) == 7:
            reordered_tfrs = torch.FloatTensor([-tfrs[0], tfrs[2], tfrs[1], tfrs[4], tfrs[3], tfrs[6], tfrs[5]])
        tensor_affineM = tfrs_to_grid_M(reordered_tfrs)
        cv2_affineM = calculate_M_from_theta(tensor_affineM, h=da_image.shape[0], w=da_image.shape[1], inversed=False).numpy()
        M_3x3 = np.vstack([cv2_affineM, [0, 0, 1]]) @ M_3x3

    M_inv = np.linalg.inv(M_3x3)
    matrix_3d = np.eye(3)
    matrix_3d[:2, :2] = M_inv[:2, :2]
    offset_3d = np.zeros(3)
    offset_3d[:2] = M_inv[:2, 2]
    kwargs_pad["cval"] = kwargs_pad.get("constant_values", 0)
    transformed_img = dask_image.ndinterp.affine_transform(
        da_image,
        matrix=matrix_3d,
        offset=offset_3d,
        output_chunks=tile_size,
        order=1,
        **kwargs_pad
    )
    return transformed_img


def resize_dask(image: da.Array, zoom=1.0, **kwargs):
    ndim = image.ndim
    matrix = np.eye(ndim) / zoom
    output_shape = list(int(dim * zoom) for dim in image.shape)
    output_shape[2] = image.shape[2]
    return dask_image.ndinterp.affine_transform(
        image,
        matrix,
        output_shape=output_shape,
        order=1,
        **kwargs
    )