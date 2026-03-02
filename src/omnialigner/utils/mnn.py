import numpy as np
import torch
import torch.nn.functional as F

from omnialigner.dtypes import Tensor_cells_N_xy, Tensor_cells_N_embed

def calculate_cdist_corr(tensor1: torch.FloatTensor, tensor2: torch.FloatTensor, eps: float=1e-8):
    """
    Batch F.cosine_similarity. Same results of:
    
    ```
    m, n = tensor1_test.shape[0], tensor2_test.shape[0]
    cos_sim_ = torch.zeros((m, n))
    for i_row in range(m):
        for j_row in range(n):
            cos_sim_[i_row, j_row] = F.cosine_similarity(tensor1_test[i_row:i_row+1], tensor2_test[j_row:j_row+1])
    
    ```

    Args:
        tensor1: [M x C] feature tensor.
        tensor2: [N x C] feature tensor.
        eps:     random noise in avoid of div 0
    
    Returns:
        [M, N] cosine similarity matrix.

    """
    norm1= tensor1.norm(dim=1, keepdim=True) + eps
    norm2= tensor2.norm(dim=1, keepdim=True) + eps
    tensor1_norm = tensor1 / norm1
    tensor2_norm = tensor2 / norm2
    euclidean_dist= torch.cdist(tensor1_norm, tensor2_norm, p=2)
    cos_sim= 1 - 0.5 * (euclidean_dist** 2)
    return cos_sim


def calculate_cdist_dist(tensor1: torch.FloatTensor, tensor2: torch.FloatTensor, p=2):
    return torch.cdist(tensor1, tensor2, p=p)


def calc_pearson_corr(pred: torch.FloatTensor, target: torch.FloatTensor, eps=1e-8):
    """
    Calculate Pearson correlation coefficient loss between predicted and target tensors.
    
    Args:
        pred: [N, C] tensor of predicted values
        target: [N, C] tensor of target values
        eps: a small value to avoid division by zero
        
    Return:
        loss: Pearson correlation coefficient for each sample [N]
    """
    pred_mean = torch.mean(pred, dim=1, keepdim=True)
    target_mean = torch.mean(target, dim=1, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    covariance = torch.sum(pred_centered * target_centered, dim=1)
    pred_std = torch.sqrt(torch.sum(pred_centered **2, dim=1) + eps)
    target_std = torch.sqrt(torch.sum(target_centered** 2, dim=1) + eps)
    std_product = pred_std * target_std
    

    pearson_corr = covariance / std_product
    return pearson_corr

def mutual_nearest_neighbors_via_matrix(dist_matrix: torch.FloatTensor, k: int=6, largest: bool=False, top_percent: float=0.1) -> np.ndarray:
    """
    Identify mutual nearest neighbors between two sets of points based on a distance matrix.

    Args:
        dist_matrix (torch.FloatTensor): A distance matrix of shape (N_A, N_B) where N_A and N_B are the number of points in sets A and B respectively.
        k (int): The number of nearest neighbors to consider for each point in both sets.
        largest (bool): If True, consider the largest distances; if False, consider the smallest distances.
        top_percent (float): The top percentage of nearest neighbors to consider.
    Returns:
        np.ndarray: An array of shape (M, 2) where each row contains the indices of a mutual nearest neighbor pair (index_in_A, index_in_B).
    """
    _, indices_A2B = torch.topk(dist_matrix, k=k, dim=1, largest=largest)
    A2B_pairs = []
    for i in range(indices_A2B.shape[0]):
        for j in range(k):
            A2B_pairs.append([i, indices_A2B[i, j].item()])
    
    _, indices_B2A = torch.topk(dist_matrix, k=k, dim=0, largest=largest)
    B2A_pairs = []
    for j in range(indices_B2A.shape[1]):
        for i in range(k):
            B2A_pairs.append([indices_B2A[i, j].item(), j])
    
    A2B_set = set(tuple(pair) for pair in A2B_pairs)
    B2A_set = set(tuple(pair) for pair in B2A_pairs)
    mnn_pairs = np.array([list(pair) for pair in A2B_set.intersection(B2A_set)])
    
    if top_percent < 1.0:
        num_top = int(len(mnn_pairs) * top_percent)
        dists = dist_matrix[mnn_pairs[:, 0], mnn_pairs[:, 1]]
        top_indices = torch.topk(dists, k=num_top, largest=largest).indices
        mnn_pairs = mnn_pairs[top_indices.cpu().numpy()]
    
    return mnn_pairs

def calculate_overlapped_mnn_pairs(
        tensor_coord_i:Tensor_cells_N_xy,
        tensor_coord_j:Tensor_cells_N_xy,
        embed_i: Tensor_cells_N_embed=None,
        embed_j: Tensor_cells_N_embed=None,
        k:int=5,
        top_percent:float=0.2
    ) -> np.ndarray:
    
    dist_spatial = calculate_cdist_dist(tensor_coord_i, tensor_coord_j, p=2)
    if embed_i is None or embed_j is None:
        dist_corr = 0.0
    else:
        dist_corr = calculate_cdist_corr(embed_i, embed_j)

    dist_overall = dist_spatial * (1-dist_corr)
    mnn_pairs = mutual_nearest_neighbors_via_matrix(dist_overall, k=k, largest=False, top_percent=top_percent)
    return mnn_pairs

def unpaired_dist(
        mnn_pairs: np.ndarray,
        dist_corr: torch.FloatTensor,
        dist_spatial: torch.FloatTensor,
    ) -> torch.FloatTensor:
    """
    Calculate the average spatial and correlation distance between two sets of coordinates using mutual nearest neighbors.

    Args:
        mnn_pairs (np.ndarray): An array of shape (M, 2) containing indices of mutual nearest neighbor pairs.
        dist_spatial (torch.FloatTensor): Spatial distance matrix between the two sets (N_ref
        dist_corr (torch.FloatTensor): Correlation distance matrix between the two sets (N_ref, N_move).
    
    Returns:
        torch.FloatTensor: A tensor containing the average spatial distance and average correlation distance.
    """
    idx_0 = mnn_pairs[:, 0]
    idx_1 = mnn_pairs[:, 1]
    avg_dist = torch.stack([torch.nanmean(dist_spatial[idx_0, idx_1]), torch.nanmean(dist_corr[idx_0, idx_1])])
    return avg_dist
