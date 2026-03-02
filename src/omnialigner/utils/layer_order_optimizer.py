"""
Layer ordering optimization for spatial transcriptomics data.

This module provides methods to refine the ordering of layers in 3D spatial datasets
by minimizing inter-layer cell distance metrics. Useful when layers are already 
roughly ordered but have local misalignments (e.g., ±5 positions).

Strategies:
1. Sliding window insertion sort: Local optimization within a fixed window
2. KNN graph + shortest path: Global optimization using graph traversal
3. Hybrid approach: Detect outliers then apply local refinement
"""

import numpy as np
import torch
from typing import List, Callable, Tuple, Optional
from tqdm import tqdm
import scanpy as sc


def calculate_layer_pair_distance(
    adata_i: sc.AnnData,
    adata_j: sc.AnnData,
    coord_key: str = "spatial",
    embed_key: Optional[str] = "X_pca",
    n_samples: int = 10000,
    device: str = "cpu"
) -> float:
    """
    Calculate distance metric between two adjacent layers.
    
    Uses combined spatial + embedding distance similar to MNN calculation.
    
    Args:
        adata_i: AnnData for layer i
        adata_j: AnnData for layer j
        coord_key: Key in .obsm for spatial coordinates
        embed_key: Key in .obsm for embeddings (None to use spatial only)
        n_samples: Number of cells to sample for calculation (to save memory)
        device: torch device
    
    Returns:
        Average combined distance between the two layers
    """
    # Sample cells if too many
    n_i = min(n_samples, adata_i.n_obs)
    n_j = min(n_samples, adata_j.n_obs)
    
    idx_i = np.random.choice(adata_i.n_obs, n_i, replace=False)
    idx_j = np.random.choice(adata_j.n_obs, n_j, replace=False)
    
    # Get coordinates
    coord_i = torch.FloatTensor(adata_i.obsm[coord_key][idx_i]).to(device)
    coord_j = torch.FloatTensor(adata_j.obsm[coord_key][idx_j]).to(device)
    
    # Spatial distance (use median of nearest neighbors)
    dist_spatial = torch.cdist(coord_i, coord_j, p=2)
    spatial_dist, _ = dist_spatial.min(dim=1)  # For each cell in i, find nearest in j
    median_spatial = spatial_dist.median().item()
    
    # Embedding distance (if available)
    if embed_key is not None and embed_key in adata_i.obsm and embed_key in adata_j.obsm:
        embed_i = torch.FloatTensor(adata_i.obsm[embed_key][idx_i]).to(device)
        embed_j = torch.FloatTensor(adata_j.obsm[embed_key][idx_j]).to(device)
        
        # Normalize embeddings
        embed_i_norm = torch.nn.functional.normalize(embed_i, dim=1, eps=1e-8)
        embed_j_norm = torch.nn.functional.normalize(embed_j, dim=1, eps=1e-8)
        
        # Cosine similarity
        cos_sim = torch.mm(embed_i_norm, embed_j_norm.T)
        max_sim, _ = cos_sim.max(dim=1)  # For each cell in i, find most similar in j
        median_sim = max_sim.median().item()
        
        # Combined metric: spatial distance weighted by dissimilarity
        combined_dist = median_spatial * (1 - median_sim)
    else:
        combined_dist = median_spatial
    
    return combined_dist


def sliding_window_insertion_sort(
    l_cells_h5ad: List[sc.AnnData],
    distance_func: Callable[[sc.AnnData, sc.AnnData], float],
    window_size: int = 7,
    verbose: bool = True
) -> List[sc.AnnData]:
    """
    Optimize layer ordering using sliding window insertion sort.
    
    For each position, considers a local window and finds the best placement
    within that window to minimize total pairwise distances.
    
    Args:
        l_cells_h5ad: List of AnnData objects (layers)
        distance_func: Function to compute distance between two layers
        window_size: Size of sliding window (should be >= 2 * error_range + 1)
        verbose: Show progress bar
    
    Returns:
        Reordered list of AnnData objects
    """
    n_layers = len(l_cells_h5ad)
    ordered = l_cells_h5ad.copy()
    
    if n_layers <= 1:
        return ordered
    
    # Iterate through positions
    iterator = tqdm(range(1, n_layers), desc="Optimizing order") if verbose else range(1, n_layers)
    
    for i in iterator:
        # Define window boundaries
        window_start = max(0, i - window_size // 2)
        window_end = min(n_layers, i + window_size // 2 + 1)
        
        if window_end - window_start <= 1:
            continue
        
        # Try inserting current element at each position in the window
        current_item = ordered[i]
        best_pos = i
        best_cost = float('inf')
        
        for insert_pos in range(window_start, window_end):
            if insert_pos == i:
                # Calculate cost for current position
                cost = 0
                if i > 0:
                    cost += distance_func(ordered[i-1], current_item)
                if i < n_layers - 1:
                    cost += distance_func(current_item, ordered[i+1])
            else:
                # Calculate cost if we move to insert_pos
                cost = 0
                # Remove current from position i
                temp = ordered[:i] + ordered[i+1:]
                # Insert at new position
                new_pos = insert_pos if insert_pos < i else insert_pos - 1
                temp.insert(new_pos, current_item)
                
                # Calculate affected distances
                if new_pos > 0:
                    cost += distance_func(temp[new_pos-1], temp[new_pos])
                if new_pos < len(temp) - 1:
                    cost += distance_func(temp[new_pos], temp[new_pos+1])
            
            if cost < best_cost:
                best_cost = cost
                best_pos = insert_pos
        
        # Perform the swap if beneficial
        if best_pos != i:
            item = ordered.pop(i)
            ordered.insert(best_pos, item)
    
    return ordered


def build_distance_matrix(
    l_cells_h5ad: List[sc.AnnData],
    distance_func: Callable[[sc.AnnData, sc.AnnData], float],
    verbose: bool = True
) -> np.ndarray:
    """
    Build pairwise distance matrix between all layers.
    
    Args:
        l_cells_h5ad: List of AnnData objects
        distance_func: Function to compute distance between two layers
        verbose: Show progress
    
    Returns:
        [N, N] distance matrix
    """
    n_layers = len(l_cells_h5ad)
    dist_matrix = np.zeros((n_layers, n_layers))
    
    iterator = tqdm(range(n_layers), desc="Building distance matrix") if verbose else range(n_layers)
    
    for i in iterator:
        for j in range(i+1, n_layers):
            dist = distance_func(l_cells_h5ad[i], l_cells_h5ad[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def greedy_nearest_neighbor_path(
    dist_matrix: np.ndarray,
    start_idx: int = 0
) -> List[int]:
    """
    Find approximate TSP solution using greedy nearest neighbor.
    
    Args:
        dist_matrix: [N, N] pairwise distance matrix
        start_idx: Starting layer index
    
    Returns:
        Ordered list of layer indices
    """
    n_layers = dist_matrix.shape[0]
    visited = [False] * n_layers
    path = [start_idx]
    visited[start_idx] = True
    
    current = start_idx
    for _ in range(n_layers - 1):
        # Find nearest unvisited neighbor
        min_dist = float('inf')
        next_node = None
        
        for j in range(n_layers):
            if not visited[j] and dist_matrix[current, j] < min_dist:
                min_dist = dist_matrix[current, j]
                next_node = j
        
        if next_node is not None:
            path.append(next_node)
            visited[next_node] = True
            current = next_node
    
    return path


def two_opt_improvement(
    path: List[int],
    dist_matrix: np.ndarray,
    max_iterations: int = 100
) -> List[int]:
    """
    Improve path using 2-opt local search.
    
    Iteratively try reversing segments of the path to reduce total cost.
    
    Args:
        path: Initial path (list of indices)
        dist_matrix: [N, N] distance matrix
        max_iterations: Maximum number of improvement iterations
    
    Returns:
        Improved path
    """
    n = len(path)
    improved_path = path.copy()
    
    def path_cost(p):
        """Calculate total adjacent-pair cost."""
        cost = 0
        for i in range(len(p) - 1):
            cost += dist_matrix[p[i], p[i+1]]
        return cost
    
    best_cost = path_cost(improved_path)
    
    for iteration in range(max_iterations):
        improved = False
        
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Try reversing segment [i:j]
                new_path = improved_path[:i] + improved_path[i:j][::-1] + improved_path[j:]
                new_cost = path_cost(new_path)
                
                if new_cost < best_cost:
                    improved_path = new_path
                    best_cost = new_cost
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return improved_path


def knn_graph_path_optimization(
    l_cells_h5ad: List[sc.AnnData],
    distance_func: Callable[[sc.AnnData, sc.AnnData], float],
    start_idx: int = 0,
    use_2opt: bool = True,
    verbose: bool = True
) -> List[sc.AnnData]:
    """
    Optimize layer ordering using KNN graph + shortest path search.
    
    Builds complete distance matrix, then finds approximate shortest path
    through all layers using greedy nearest neighbor + 2-opt refinement.
    
    Args:
        l_cells_h5ad: List of AnnData objects
        distance_func: Function to compute distance between two layers
        start_idx: Starting layer index
        use_2opt: Whether to apply 2-opt improvement
        verbose: Show progress
    
    Returns:
        Reordered list of AnnData objects
    """
    # Build distance matrix
    dist_matrix = build_distance_matrix(l_cells_h5ad, distance_func, verbose=verbose)
    
    # Find path using greedy nearest neighbor
    if verbose:
        print("Finding greedy path...")
    path = greedy_nearest_neighbor_path(dist_matrix, start_idx)
    
    # Improve with 2-opt
    if use_2opt:
        if verbose:
            print("Applying 2-opt improvement...")
        path = two_opt_improvement(path, dist_matrix)
    
    # Reorder layers
    ordered = [l_cells_h5ad[i] for i in path]
    
    return ordered


def hybrid_optimization(
    l_cells_h5ad: List[sc.AnnData],
    distance_func: Callable[[sc.AnnData, sc.AnnData], float],
    outlier_threshold: float = 2.0,
    window_size: int = 7,
    verbose: bool = True
) -> List[sc.AnnData]:
    """
    Hybrid approach: Detect outliers using global distances, then apply local refinement.
    
    Strategy:
    1. Compute distances between adjacent layers in current order
    2. Identify outliers (distances > threshold * median)
    3. For outlier regions, search wider neighborhood for better placement
    4. Apply sliding window insertion sort for final refinement
    
    Args:
        l_cells_h5ad: List of AnnData objects
        distance_func: Function to compute distance between two layers
        outlier_threshold: Multiplier of median distance to detect outliers
        window_size: Size of window for local optimization
        verbose: Show progress
    
    Returns:
        Reordered list of AnnData objects
    """
    n_layers = len(l_cells_h5ad)
    ordered = l_cells_h5ad.copy()
    
    if n_layers <= 2:
        return ordered
    
    # Step 1: Compute adjacent distances
    if verbose:
        print("Computing adjacent distances...")
    adjacent_dists = []
    for i in range(n_layers - 1):
        dist = distance_func(ordered[i], ordered[i+1])
        adjacent_dists.append(dist)
    
    adjacent_dists = np.array(adjacent_dists)
    median_dist = np.median(adjacent_dists)
    
    # Step 2: Identify outliers
    outliers = adjacent_dists > outlier_threshold * median_dist
    outlier_positions = np.where(outliers)[0]
    
    if verbose:
        print(f"Found {len(outlier_positions)} outlier positions: {outlier_positions}")
        print(f"Median distance: {median_dist:.4f}")
        print(f"Outlier distances: {adjacent_dists[outliers]}")
    
    # Step 3: Fix outliers with wider search
    for pos in outlier_positions:
        # The issue is between pos and pos+1
        # Try swapping pos+1 with neighbors in a wider window
        target_idx = pos + 1
        search_start = max(0, target_idx - 10)
        search_end = min(n_layers, target_idx + 10)
        
        current_item = ordered[target_idx]
        best_pos = target_idx
        best_cost = adjacent_dists[pos]
        
        if verbose:
            print(f"  Fixing outlier at position {pos}->{pos+1}, searching [{search_start}, {search_end})")
        
        for new_pos in range(search_start, search_end):
            if new_pos == target_idx:
                continue
            
            # Try moving to new_pos
            temp = ordered[:target_idx] + ordered[target_idx+1:]
            insert_pos = new_pos if new_pos < target_idx else new_pos - 1
            temp.insert(insert_pos, current_item)
            
            # Calculate cost for affected pairs
            cost = 0
            if insert_pos > 0:
                cost += distance_func(temp[insert_pos-1], temp[insert_pos])
            if insert_pos < len(temp) - 1:
                cost += distance_func(temp[insert_pos], temp[insert_pos+1])
            
            if cost < best_cost:
                best_cost = cost
                best_pos = new_pos
        
        # Apply the move
        if best_pos != target_idx:
            if verbose:
                print(f"    Moving from {target_idx} to {best_pos}, cost: {adjacent_dists[pos]:.4f} -> {best_cost:.4f}")
            item = ordered.pop(target_idx)
            insert_pos = best_pos if best_pos < target_idx else best_pos - 1
            ordered.insert(insert_pos, item)
    
    # Step 4: Final refinement with sliding window
    if verbose:
        print("Applying final sliding window refinement...")
    ordered = sliding_window_insertion_sort(ordered, distance_func, window_size, verbose=False)
    
    return ordered


def evaluate_ordering(
    l_cells_h5ad: List[sc.AnnData],
    distance_func: Callable[[sc.AnnData, sc.AnnData], float],
    verbose: bool = True
) -> dict:
    """
    Evaluate the quality of layer ordering.
    
    Args:
        l_cells_h5ad: List of AnnData objects
        distance_func: Function to compute distance between two layers
        verbose: Print results
    
    Returns:
        Dictionary with evaluation metrics
    """
    n_layers = len(l_cells_h5ad)
    
    if n_layers <= 1:
        return {"total_cost": 0, "mean_dist": 0, "median_dist": 0, "max_dist": 0}
    
    # Compute adjacent distances
    adjacent_dists = []
    for i in range(n_layers - 1):
        dist = distance_func(l_cells_h5ad[i], l_cells_h5ad[i+1])
        adjacent_dists.append(dist)
    
    adjacent_dists = np.array(adjacent_dists)
    
    metrics = {
        "total_cost": adjacent_dists.sum(),
        "mean_dist": adjacent_dists.mean(),
        "median_dist": np.median(adjacent_dists),
        "max_dist": adjacent_dists.max(),
        "min_dist": adjacent_dists.min(),
        "std_dist": adjacent_dists.std()
    }
    
    if verbose:
        print("=== Layer Ordering Evaluation ===")
        print(f"Total cost: {metrics['total_cost']:.4f}")
        print(f"Mean distance: {metrics['mean_dist']:.4f}")
        print(f"Median distance: {metrics['median_dist']:.4f}")
        print(f"Distance range: [{metrics['min_dist']:.4f}, {metrics['max_dist']:.4f}]")
        print(f"Std deviation: {metrics['std_dist']:.4f}")
    
    return metrics
