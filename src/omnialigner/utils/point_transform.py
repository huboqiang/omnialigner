from typing import Tuple
import torch
import numpy as np
import faiss
import torch.nn.functional as F
from omnialigner.dtypes import Tensor_cv2_affine_M, Tensor_kpts_N_xky_knn, Tensor_kpts_N_xy, Tensor_kpts_N_xy_raw, Np_image_HWC, Np_kpts_N_yx_raw, Tensor_grid_2d


def raw_landmarks_to_scaled(
        tensor_landmarks_raw: Tensor_kpts_N_xy_raw,
        image: Np_image_HWC
    ) -> Tensor_kpts_N_xy:
    """
    Args:
        tensor_landmarks_raw:  [x, y] numpy coords
        image:             [H, W] numpy image
    Return:
        scaled_landmarks:  [x/W, y/H] numpy coords
    """
    nrow, ncol = image.shape[0], image.shape[1]
    scaled_landmarks = tensor_landmarks_raw.clone()
    
    scaled_landmarks[:, 0] /= ncol
    scaled_landmarks[:, 1] /= nrow
    return scaled_landmarks




def scaled_landmarks_to_raw(
        landmark_scaled: Tensor_kpts_N_xy, 
        image: Np_image_HWC | torch.FloatTensor,
    ) -> Tensor_kpts_N_xy_raw:
    """
    Args:
        scaled_landmarks:  [x/W, y/H] torch coords
        image:             [H, W] torch image
    Return:
        landmarks_raw:     [x, y] torch coords
    """
    nrow, ncol = image.shape[0], image.shape[1]
    landmarks_raw = torch.empty_like(landmark_scaled).float()
    landmarks_raw[:, 0] = landmark_scaled[:, 0] * ncol
    landmarks_raw[:, 1] = landmark_scaled[:, 1] * nrow
    return landmarks_raw

def raw_landmarks_to_padded(
        tensor_landmarks_raw: Tensor_kpts_N_xy_raw, 
        ratio: float=1, 
        padded_size: Tuple[int, int, int, int]=[0, 0, 0, 0]
    ) -> Tensor_kpts_N_xy_raw:
    """
    Args:
        tensor_landmarks_raw:  [x, y] torch coords
        ratio:             Default 1.
        padded_size:       Default None. 
                            See dask.array.pad (https://docs.dask.org/en/stable/generated/dask.array.pad.html)
                            
    Return:
        tensor_landmarks_pad:  [x, y] torch coords
    """
    tensor_landmarks_pad = tensor_landmarks_raw.clone() * ratio
    tensor_landmarks_pad[:, 0] = tensor_landmarks_pad[:, 0] + padded_size[1]
    tensor_landmarks_pad[:, 1] = tensor_landmarks_pad[:, 1] + padded_size[3]
    return tensor_landmarks_pad


def padded_landmarks_to_raw(
        np_landmarks_pad: Np_kpts_N_yx_raw, 
        ratio: float=1,
        padded_size: Tuple[int, int, int, int]=[0, 0, 0, 0]
    ) -> Np_kpts_N_yx_raw:
    """
    Args:
        np_landmarks_pad:  [x, y] numpy coords
        ratio:             Default 1.
        padded_size:       Default None. [pad_left, pad_right, pad_up, pad_down]
                            
    Return:
        np_landmarks_raw:  [x, y] numpy coords
    """
    np_landmarks_raw = np_landmarks_pad.clone()
    pad_left, _, pad_up, _ = padded_size
    
    np_landmarks_raw[:, 0] = np_landmarks_raw[:, 0] - pad_left
    np_landmarks_raw[:, 1] = np_landmarks_raw[:, 1] - pad_up
    np_landmarks_raw = np_landmarks_raw / ratio
    return np_landmarks_raw




def transfer_landmarks_inv(
        tensor_target_landmark_raw: Tensor_kpts_N_xy_raw, 
        grid: Tensor_grid_2d
    ) -> Tensor_kpts_N_xy_raw:
    """
    Args:
        tensor_target_landmark_raw:    [x, y] torch coords
        grid:                      [1, H, W, 2], See torch.nn.functional.grid_sample
    Return:
        tensor_target_landmark_inv:    Tensor_kpts_N_xy_raw, torch coords for grid's inverse coords
    """
    if tensor_target_landmark_raw.shape[0] == 0:
        return tensor_target_landmark_raw
    
    tensor_coords_ = tensor_target_landmark_raw.clone().int()
    n_row, n_col = grid.shape[1], grid.shape[2]
    is_valid_row = (tensor_coords_[:, 1] >= 0) * (tensor_coords_[:, 1] < n_row)
    is_valid_col = (tensor_coords_[:, 0] >= 0) * (tensor_coords_[:, 0] < n_col)
    is_valid = is_valid_col * is_valid_row
    tensor_xy_source_ = (grid.detach()[0, tensor_coords_[is_valid, 1], tensor_coords_[is_valid, 0], :] + 1) / 2
    tensor_xy_source_[:, 0] = tensor_xy_source_[:, 0] * n_col
    tensor_xy_source_[:, 1] = tensor_xy_source_[:, 1] * n_row
    tensor_target_landmark_inv = -1*torch.ones_like(tensor_coords_).float()
    tensor_target_landmark_inv[is_valid, :] = tensor_xy_source_
    return tensor_target_landmark_inv


def transform_keypoints(
        landmarks_raw: Tensor_kpts_N_xy_raw, 
        meta: Tensor_cv2_affine_M
    ) -> Tensor_kpts_N_xy_raw:
    """
    Args:
        landmarks_raw:  [x, y] torch coords
        meta:           Tensor cv2 M matrix for affine transform
    Return:
        landmarks_raw_transformed:  [x, y] torch coords after affine transform 
    """
    landmarks_raw_transformed = torch.empty_like(landmarks_raw).float()
    landmarks_raw_transformed[:, :2] = torch.mm(landmarks_raw[:, :2], meta[:2, :2].T) + meta[:2, 2]
    return landmarks_raw_transformed




def warp_landmarks_grid(
        kpts: Tensor_kpts_N_xy, 
        grid: Tensor_grid_2d, 
        use_faiss: bool=False,
        k: int=1
    ) -> Tensor_kpts_N_xky_knn:
    if use_faiss:
        return warp_landmark_grid_faiss(kpts, grid, k)
    
    return warp_landmark_grid(kpts, grid, k)


def warp_landmark_grid(
        landmarks: Tensor_kpts_N_xy, 
        grid: Tensor_grid_2d, 
        k: int=1
    ) -> Tensor_kpts_N_xky_knn:
    """
    Forward(landmarks) using grid
    
    Args:
        landmarks (Tensor_kpts_N_xy): input landmark , [N, 2] scaled to [0, 1]
        grid      (torch.Tensor): grid used by `F.grid_sample(image_src, grid)`
        k (int):  knn. default as 1

    
    Returns:
        tensor_target:  Tensor_kpts_N_xky_knn: [N, k, 2]，
    """
    _, H, W, _ = grid.shape
    source_coords = grid.view(-1, 2)
    target_coords = F.affine_grid(torch.eye(3)[:2, :].unsqueeze(0), (1, 1, H, W), align_corners=False).view(-1, 2).to(source_coords.device)
    
    tensor_coords = 2 * landmarks - 1
    batch_size = 1024  # Adjust batch size based on your memory constraints
    indices_list = []
    for i in range(0, tensor_coords.size(0), batch_size):
        batch_tensor_coords = tensor_coords[i:i + batch_size]
        distances = torch.cdist(batch_tensor_coords.unsqueeze(0), source_coords.unsqueeze(0)).squeeze(0)
        _, batch_indices = torch.topk(distances, k, largest=False)
        indices_list.append(batch_indices)

    indices = torch.cat(indices_list, dim=0)
    tensor_sources = source_coords[indices]
    tensor_target = 0.5 * (target_coords[indices] + 1)

    # keep grad in grid
    tensor_target = tensor_target - (tensor_sources - tensor_sources.detach())
    return tensor_target

def warp_landmark_grid_faiss(
        landmarks: Tensor_kpts_N_xy, 
        grid: Tensor_grid_2d, 
        k: int=1
    ) -> Tensor_kpts_N_xky_knn:
    """
    Forward(landmarks) using grid with Faiss acceleration
    Args:
        landmarks (Tensor_kpts_N_xy): input landmark , [N, 2] scaled to [0, 1]
        grid      (Tensor_grid_2d): grid used by `F.grid_sample(image_src, grid)`
        k (int):  knn. default as 1
    Returns:
        tensor_target:  Tensor_kpts_N_xky_knn, [N, k, 2]， scaled to [0, 1]
    """

    _, H, W, _ = grid.shape
    source_coords = grid.view(-1, 2)
    target_coords = F.affine_grid(torch.eye(3)[:2, :].unsqueeze(0), (1, 1, H, W), align_corners=False).view(-1, 2).to(source_coords.device)
    
    tensor_coords = 2 * landmarks - 1
    index = faiss.IndexFlatL2(2)
    index.add(source_coords.detach().cpu().numpy().astype(np.float32))  # Ensure float32


    _, indices = index.search(tensor_coords.detach().cpu().numpy().astype(np.float32).copy(), k)  # Ensure float32
    indices = torch.from_numpy(indices).to(landmarks.device)

    tensor_sources = source_coords[indices]
    tensor_target = 0.5 * (target_coords[indices] + 1)

    tensor_target = tensor_target - (tensor_sources - tensor_sources.detach())
    return tensor_target


def calculate_tre(
        source_landmarks: Np_kpts_N_yx_raw, 
        target_landmarks: Np_kpts_N_yx_raw
    ) -> float:
    """
    Calculate TRE between source and target landmarks
    Args:
        source_landmarks:  Np_kpts_N_yx_raw 
        target_landmarks:  Np_kpts_N_yx_raw 
    Returns:
        tre:  float
    """
    n = min(source_landmarks.shape[0], target_landmarks.shape[0])
    tre = np.sqrt(np.square(source_landmarks[0:n, 0] - target_landmarks[0:n, 0]) +\
                  np.square(source_landmarks[0:n, 1] - target_landmarks[0:n, 1]))
    return tre

def calculate_rtre(
        source_landmarks: Np_kpts_N_yx_raw, 
        target_landmarks: Np_kpts_N_yx_raw, 
        image_diagonal: float
    ) -> float:
    """
    Calculate R-TRE between source and target landmarks
    Args:
        source_landmarks:  Np_kpts_N_yx_raw 
        target_landmarks:  Np_kpts_N_yx_raw 
        image_diagonal:    float
    Returns:
        rtre:  float
    """
    tre = calculate_tre(source_landmarks, target_landmarks)
    rtre = tre / image_diagonal
    return rtre