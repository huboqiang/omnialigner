"""
Optimized mutual nearest neighbors (MNN) implementation for large-scale data.

This module provides memory-efficient implementations that avoid computing full N*M distance matrices,
suitable for handling tens of thousands of samples.

Key optimizations:
1. Chunk-based processing to limit memory usage
2. Early spatial filtering using KNN to reduce candidate pairs
3. Incremental distance computation instead of full matrices
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

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


def find_knn_chunked(
    query_coords: torch.FloatTensor,
    reference_coords: torch.FloatTensor,
    k: int,
    chunk_size: int = 1000,
    largest: bool = False
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    Find k-nearest neighbors in a memory-efficient chunked manner.
    
    Args:
        query_coords: [N_query, D] coordinates
        reference_coords: [N_ref, D] coordinates
        k: number of nearest neighbors
        chunk_size: process query points in chunks to save memory
        largest: if True, find k largest distances; if False, find k smallest
    
    Returns:
        distances: [N_query, k] distances to k-nearest neighbors
        indices: [N_query, k] indices of k-nearest neighbors in reference_coords
    """
    n_query = query_coords.shape[0]
    # device = query_coords.device
    
    all_distances = []
    all_indices = []
    
    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_chunk = query_coords[start_idx:end_idx]
        
        # Compute distances for this chunk
        dist_chunk = torch.cdist(query_chunk, reference_coords, p=2)
        
        # Find top-k for this chunk
        topk_dists, topk_indices = torch.topk(
            dist_chunk, k=min(k, reference_coords.shape[0]), 
            dim=1, largest=largest
        )
        
        all_distances.append(topk_dists)
        all_indices.append(topk_indices)
    
    distances = torch.cat(all_distances, dim=0)
    indices = torch.cat(all_indices, dim=0)
    
    return distances, indices


def calculate_distance_for_pairs(
    pairs: np.ndarray,
    coord_i: torch.FloatTensor,
    coord_j: torch.FloatTensor,
    embed_i: Optional[torch.FloatTensor]=None,
    embed_j: Optional[torch.FloatTensor]=None,
    chunk_size: int = 5000,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute spatial distance, correlation distance, and their weighted combination for given index pairs.

    Args:
        pairs: [M, 2] array of index pairs (idx_i, idx_j)
        coord_i: [N_i, 2] coordinates for set i
        coord_j: [N_j, 2] coordinates for set j
        embed_i: [N_i, C] embeddings for set i (optional)
        embed_j: [N_j, C] embeddings for set j (optional)
        chunk_size: chunk size for processing
        alpha: weight for spatial distance
        beta: weight for correlation distance

    Returns:
        dist_spatial: [M] spatial distance for each pair
        dist_corr: [M] correlation distance for each pair
        dist_combined: [M] combined distance (alpha * spatial + beta * correlation)
    """
    n_pairs = pairs.shape[0]
    all_dist_spatial = []
    all_dist_corr = []
    
    for start_idx in range(0, n_pairs, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pairs)
        pair_chunk = pairs[start_idx:end_idx]
        idx_i = pair_chunk[:, 0]
        idx_j = pair_chunk[:, 1]
        coords_i_chunk = coord_i[idx_i]
        coords_j_chunk = coord_j[idx_j]
        dist_spatial_chunk = torch.norm(coords_i_chunk - coords_j_chunk, dim=1)
        all_dist_spatial.append(dist_spatial_chunk)
        if embed_i is None or embed_j is None:
            dist_corr_chunk = torch.zeros_like(dist_spatial_chunk)
        else:
            embed_i_chunk = embed_i[idx_i]
            embed_j_chunk = embed_j[idx_j]
            
            embed_i_norm = F.normalize(embed_i_chunk, dim=1, eps=1e-8)
            embed_j_norm = F.normalize(embed_j_chunk, dim=1, eps=1e-8)
            cos_sim = (embed_i_norm * embed_j_norm).sum(dim=1)
            
            dist_corr_chunk = 1.0 - cos_sim.clamp(-1.0, 1.0)
        
        all_dist_corr.append(dist_corr_chunk)
    
    dist_spatial = torch.cat(all_dist_spatial, dim=0)
    dist_corr = torch.cat(all_dist_corr, dim=0)
    
    dist_combined = alpha * dist_spatial + beta * dist_corr
    
    return dist_spatial, dist_corr, dist_combined


def mutual_nearest_neighbors_fast(
    coord_i: torch.FloatTensor,
    coord_j: torch.FloatTensor,
    embed_i: Optional[torch.FloatTensor],
    embed_j: Optional[torch.FloatTensor],
    k: int = 6,
    spatial_k_factor: int = 3,
    chunk_size: int = 1000,
    top_percent: float = 0.9
) -> np.ndarray:
    """
    Fast mutual nearest neighbor search using spatial pre-filtering.
    
    Strategy:
    1. First, find spatial k-nearest neighbors (using spatial_k_factor * k to get more candidates)
    2. Then, compute combined distance (spatial + correlation) only for these candidates
    3. Finally, find mutual nearest neighbors from refined candidates
    
    This avoids computing the full N*M distance matrix.
    
    Args:
        coord_i: [N_i, 2] coordinates for set i
        coord_j: [N_j, 2] coordinates for set j
        embed_i: [N_i, C] embeddings for set i (optional)
        embed_j: [N_j, C] embeddings for set j (optional)
        k: number of nearest neighbors for MNN
        spatial_k_factor: multiplier for initial spatial filtering (use spatial_k_factor * k candidates)
        chunk_size: chunk size for memory-efficient processing
        top_percent: keep top percentage of MNN pairs by distance
    
    Returns:
        mnn_pairs: [M, 2] array of mutual nearest neighbor pairs
    """
    device = coord_i.device
    n_i = coord_i.shape[0]
    n_j = coord_j.shape[0]
    
    # Determine spatial k for pre-filtering
    # IMPROVEMENT: Increase candidate pool for better recall
    effective_spatial_k = max(spatial_k_factor * k, k * 5)
    spatial_k = min(effective_spatial_k, n_j)
    
    # Step 1: Find spatial k-nearest neighbors from i to j
    _, spatial_indices_i2j = find_knn_chunked(
        coord_i, coord_j, k=spatial_k, chunk_size=chunk_size, largest=False
    )
    
    # Step 2: Find spatial k-nearest neighbors from j to i
    spatial_k_ji = min(effective_spatial_k, n_i)
    _, spatial_indices_j2i = find_knn_chunked(
        coord_j, coord_i, k=spatial_k_ji, chunk_size=chunk_size, largest=False
    )
    
    # Step 3: Create candidate pairs from spatial neighbors
    candidate_pairs_i2j = []
    for i in range(n_i):
        for kk in range(spatial_indices_i2j.shape[1]):
            j = spatial_indices_i2j[i, kk].item()
            candidate_pairs_i2j.append([i, j])
    
    candidate_pairs_j2i = []
    for j in range(n_j):
        for kk in range(spatial_indices_j2i.shape[1]):
            i = spatial_indices_j2i[j, kk].item()
            candidate_pairs_j2i.append([i, j])
    
    candidate_pairs_i2j = np.array(candidate_pairs_i2j)
    candidate_pairs_j2i = np.array(candidate_pairs_j2i)
    
    # Step 4: Compute combined distances for candidate pairs only
    if len(candidate_pairs_i2j) == 0 or len(candidate_pairs_j2i) == 0:
        return np.array([]).reshape(0, 2)
    
    _, _, dists_i2j = calculate_distance_for_pairs(
        candidate_pairs_i2j, coord_i, coord_j, embed_i, embed_j, chunk_size=chunk_size
    )
    
    _, _, dists_j2i = calculate_distance_for_pairs(
        candidate_pairs_j2i, coord_i, coord_j, embed_i, embed_j, chunk_size=chunk_size
    )
    
    # Step 5: For each point in i, select k nearest from candidates
    i2j_dict = {}
    for idx, (i, j) in enumerate(candidate_pairs_i2j):
        if i not in i2j_dict:
            i2j_dict[i] = []
        i2j_dict[i].append((j, dists_i2j[idx].item()))
    
    A2B_pairs = []
    for i in i2j_dict:
        # Sort by distance and take top k
        sorted_neighbors = sorted(i2j_dict[i], key=lambda x: x[1])[:k]
        for j, _ in sorted_neighbors:
            A2B_pairs.append([i, j])
    
    # Step 6: For each point in j, select k nearest from candidates
    j2i_dict = {}
    for idx, (i, j) in enumerate(candidate_pairs_j2i):
        if j not in j2i_dict:
            j2i_dict[j] = []
        j2i_dict[j].append((i, dists_j2i[idx].item()))
    
    B2A_pairs = []
    for j in j2i_dict:
        # Sort by distance and take top k
        sorted_neighbors = sorted(j2i_dict[j], key=lambda x: x[1])[:k]
        for i, _ in sorted_neighbors:
            B2A_pairs.append([i, j])
    
    # Step 7: Find mutual nearest neighbors
    A2B_set = set(tuple(pair) for pair in A2B_pairs)
    B2A_set = set(tuple(pair) for pair in B2A_pairs)
    mnn_pairs = np.array([list(pair) for pair in A2B_set.intersection(B2A_set)])
    
    if len(mnn_pairs) == 0:
        return mnn_pairs
    
    # Step 8: Keep only top percentage by distance
    if top_percent < 1.0:
        # Recompute distances for final MNN pairs
        _, _, mnn_dists = calculate_distance_for_pairs(
            mnn_pairs, coord_i, coord_j, embed_i, embed_j, chunk_size=chunk_size
        )
        
        num_top = max(1, int(len(mnn_pairs) * top_percent))
        top_indices = torch.topk(mnn_dists, k=num_top, largest=False).indices
        mnn_pairs = mnn_pairs[top_indices.cpu().numpy()]
    
    return mnn_pairs


def calculate_overlapped_mnn_pairs(
    tensor_coord_i: Tensor_cells_N_xy,
    tensor_coord_j: Tensor_cells_N_xy,
    embed_i: Tensor_cells_N_embed = None,
    embed_j: Tensor_cells_N_embed = None,
    k: int = 5,
    top_percent: float = 0.9,
    spatial_k_factor: int = 3,
    chunk_size: int = 1000
) -> np.ndarray:
    """
    Memory-efficient calculation of overlapped mutual nearest neighbor pairs.
    
    This optimized version avoids computing the full N*M distance matrix by:
    1. Using spatial coordinates to pre-filter candidates
    2. Processing data in chunks
    3. Computing combined distances only for candidate pairs
    
    Suitable for handling tens of thousands of samples.
    
    Args:
        tensor_coord_i: [N_i, 2] coordinates for cells in image i
        tensor_coord_j: [N_j, 2] coordinates for cells in image j
        embed_i: [N_i, C] embeddings for cells in image i (optional)
        embed_j: [N_j, C] embeddings for cells in image j (optional)
        k: number of nearest neighbors to consider for MNN
        top_percent: keep top percentage of MNN pairs by combined distance
        spatial_k_factor: multiplier for spatial pre-filtering (larger = more candidates, more accurate but slower)
        chunk_size: chunk size for processing (adjust based on available memory)
    
    Returns:
        mnn_pairs: [M, 2] array where each row contains indices (idx_i, idx_j) of a MNN pair
    
    Memory complexity: O(N * k * spatial_k_factor) instead of O(N * M)
    Time complexity: O(N * k * spatial_k_factor * C) where C is embedding dimension
    """
    return mutual_nearest_neighbors_fast(
        coord_i=tensor_coord_i,
        coord_j=tensor_coord_j,
        embed_i=embed_i,
        embed_j=embed_j,
        k=k,
        spatial_k_factor=spatial_k_factor,
        chunk_size=chunk_size,
        top_percent=top_percent
    )
