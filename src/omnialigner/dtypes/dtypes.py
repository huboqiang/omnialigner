from typing import List, Tuple
from enum import Enum
import torch
import numpy as np
import dask.array as da

""" 
    | Name                  | DataType         | DataSize     | Description                      |
    |-----------------------|------------------|--------------|----------------------------------|
    | Tensor_tfrs           | torch.FloatTensor| [7]          | [theta, tx, ty, sx, sy, fx, fy]  |
    | Tensor_trs            | torch.FloatTensor| [5]          | [theta, tx, ty, sx, sy]          |
    | Np_cv2_affine_M       | np.ndarray       | [2, 3]       | OpenCV affine matrix             |
    | Tensor_cv2_affine_M   | np.ndarray       | [2, 3]       | OpenCV affine matrix in pytorch  |
    | Tensor_grid_M         | torch.FloatTensor| [2, 3]       | F.affine_grid affine matrix      |
    | Tensor_grid_2d        | torch.FloatTensor| [N, H, W, 2] | grid for torch F.grid_sample     |
    | Tensor_disp_2d        | torch.FloatTensor| [N, H, W, 2] | disp_field for non-rigid         |
    | Np_disp_2d            | np.ndarray       | [N, H, W, 2] | disp_field for non-rigid         |
    | Np_image_HWC          | np.ndarray       | [H, W, C]    | Image array in HWC format        |
    | Np_image_Mask         | np.ndarray       | [H, W]       | Image array in HW mask format    |
    | Tensor_image_NCHW     | torch.FloatTensor| [N, C, H, W] | Image tensor in NCHW format      |
    | Dask_image_HWC        | da.Array         | [N, H, W, C] | Dask array in HWC format         |
    | Dask_image_NCHW       | da.Array         | [N, C, H, W] | Dask array in NCHW format        |
    | Tensor_kpts_N_xy      | torch.FloatTensor| [N, 2]       | kpt tensor scaled into [0, 1]    |
    | Tensor_kpts_N_xy_raw  | torch.FloatTensor| [N, 2]       | kpt tensor raw                   |
    | Tensor_kpts_N_xky_knn | torch.FloatTensor| [N, K, 2]    | KNN nearest kpt scaled [0, 1]    |
    | Np_kpts_N_yx_raw      | np.ndarray       | [N, 2]       | kpt array raw                    |
    | Np_kpts_N_yx          | np.ndarray       | [N, 2]       | kpt array scaled into [0, 1]     |
    | Tensor_kpts_N_embed   | torch.FloatTensor| [N, C]       | Embed keypoints tensor for adata |
    | Tuple_bbox            | Tuple[int, ...]  | [ int x4 ]   | Bbox(x_beg, y_beg, x_end, y_end) |
    | ConfigFile            | str              | -            | Configuration file path          |
"""

Tensor_tfrs = torch.FloatTensor
Tensor_trs = torch.FloatTensor

Np_cv2_affine_M = np.ndarray
Tensor_cv2_affine_M = np.ndarray
Tensor_grid_M = torch.FloatTensor

Tensor_grid_2d = torch.FloatTensor
Tensor_disp_2d = torch.FloatTensor
Np_disp_2d = np.ndarray

Tensor_image_NCHW = torch.FloatTensor
Dask_image_NCHW = da.Array
Np_image_HWC = np.ndarray
Dask_image_HWC = da.Array
Np_image_Mask = np.ndarray

Tensor_kpts_N_xy = torch.FloatTensor
Tensor_kpts_N_xy_raw = torch.FloatTensor
Tensor_kpts_N_xky_knn = torch.FloatTensor
Tensor_kpt_pair = Tuple[Tensor_kpts_N_xy, Tensor_kpts_N_xy]
Tensor_Layer_kpt_pair = List[Tensor_kpt_pair]
Tensor_l_kpt_pair = List[Tensor_Layer_kpt_pair]
Np_kpts_N_yx_raw = np.ndarray
Np_kpts_N_yx = np.ndarray

Tensor_kpts_N_embed = torch.FloatTensor
Tuple_bbox = Tuple[float, float, float, float]  # (x_beg, y_beg, x_end, y_end)
ConfigFile = str

class DataType(Enum):
    RAW = 0
    GRAY = 1
    CLS = 2
    SUB = 3
    RMAF = 4
    MASK = 5

    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        if isinstance(other, DataType):
            return self.value == other.value
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
tag_map = {
    "SUB": DataType.SUB,
    "CLS": DataType.CLS,
    "RAW": DataType.RAW,
    "GRAY": DataType.GRAY,
    "RMAF": DataType.RMAF,
    "MASK": DataType.MASK
}
