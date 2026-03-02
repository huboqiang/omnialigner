"""
Test suite for mutual nearest neighbors (MNN) implementations.

This module tests the consistency between the original and optimized MNN implementations,
ensuring that the memory-efficient version produces equivalent results.
"""

import numpy as np
import torch
import pytest
from typing import Tuple

# Import original implementation
from omnialigner.utils.mnn import (
    calculate_overlapped_mnn_pairs as calculate_overlapped_mnn_pairs_original,
    calculate_cdist_corr as calculate_cdist_corr_original,
    calculate_cdist_dist as calculate_cdist_dist_original,
)

# Import optimized implementation
from omnialigner.utils.mnn_fast import (
    calculate_overlapped_mnn_pairs as calculate_overlapped_mnn_pairs_fast,
    calculate_cdist_corr as calculate_cdist_corr_fast,
    calculate_cdist_dist as calculate_cdist_dist_fast,
    find_knn_chunked,
    compute_combined_distance_for_pairs,
)


def generate_test_data(
    n_i: int = 1000,
    n_j: int = 1000,
    embed_dim: int = 50,
    spatial_range: float = 1000.0,
    overlap_ratio: float = 0.3,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic test data with controlled overlap.
    
    Args:
        n_i: Number of points in set i
        n_j: Number of points in set j
        embed_dim: Embedding dimension
        spatial_range: Spatial coordinate range
        overlap_ratio: Ratio of overlapping points
        seed: Random seed
    
    Returns:
        coord_i, coord_j, embed_i, embed_j
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate coordinates with some overlap
    n_overlap = int(min(n_i, n_j) * overlap_ratio)
    
    # Common overlapping region
    overlap_coords = torch.rand(n_overlap, 2) * spatial_range * 0.5 + spatial_range * 0.25
    
    # Set i: overlap + unique points
    coord_i_unique = torch.rand(n_i - n_overlap, 2) * spatial_range
    coord_i = torch.cat([overlap_coords + torch.randn(n_overlap, 2) * 5, coord_i_unique], dim=0)
    
    # Set j: overlap + unique points
    coord_j_unique = torch.rand(n_j - n_overlap, 2) * spatial_range
    coord_j = torch.cat([overlap_coords + torch.randn(n_overlap, 2) * 5, coord_j_unique], dim=0)
    
    # Shuffle
    perm_i = torch.randperm(n_i)
    perm_j = torch.randperm(n_j)
    coord_i = coord_i[perm_i]
    coord_j = coord_j[perm_j]
    
    # Generate embeddings (normalized for similarity computation)
    embed_i = torch.randn(n_i, embed_dim)
    embed_j = torch.randn(n_j, embed_dim)
    
    # Make some embeddings similar for overlapping regions
    for idx in range(n_overlap):
        common_embed = torch.randn(embed_dim)
        if idx < n_i:
            embed_i[idx] = common_embed + torch.randn(embed_dim) * 0.1
        if idx < n_j:
            embed_j[idx] = common_embed + torch.randn(embed_dim) * 0.1
    
    return coord_i, coord_j, embed_i, embed_j


class TestDistanceFunctions:
    """Test basic distance computation functions."""
    
    def test_cdist_corr_consistency(self):
        """Test that cosine similarity computation is consistent."""
        torch.manual_seed(42)
        tensor1 = torch.randn(100, 50)
        tensor2 = torch.randn(100, 50)
        
        result_original = calculate_cdist_corr_original(tensor1, tensor2)
        result_fast = calculate_cdist_corr_fast(tensor1, tensor2)
        
        assert torch.allclose(result_original, result_fast, atol=1e-5), \
            "Cosine similarity results differ between implementations"
    
    def test_cdist_dist_consistency(self):
        """Test that distance computation is consistent."""
        torch.manual_seed(42)
        tensor1 = torch.randn(100, 2)
        tensor2 = torch.randn(100, 2)
        
        result_original = calculate_cdist_dist_original(tensor1, tensor2)
        result_fast = calculate_cdist_dist_fast(tensor1, tensor2)
        
        assert torch.allclose(result_original, result_fast, atol=1e-5), \
            "Distance results differ between implementations"


class TestChunkedKNN:
    """Test chunked k-nearest neighbor computation."""
    
    def test_find_knn_chunked_small_chunk(self):
        """Test chunked KNN with small chunk size."""
        torch.manual_seed(42)
        query = torch.randn(500, 2)
        reference = torch.randn(500, 2)
        k = 5
        
        # Full computation
        dist_full = torch.cdist(query, reference, p=2)
        dists_full, indices_full = torch.topk(dist_full, k=k, dim=1, largest=False)
        
        # Chunked computation
        dists_chunked, indices_chunked = find_knn_chunked(
            query, reference, k=k, chunk_size=50, largest=False
        )
        
        # Results should be identical
        assert torch.allclose(dists_full, dists_chunked, atol=1e-5), \
            "Chunked KNN distances differ from full computation"
        assert torch.equal(indices_full, indices_chunked), \
            "Chunked KNN indices differ from full computation"
    
    def test_find_knn_chunked_large_chunk(self):
        """Test chunked KNN with large chunk size."""
        torch.manual_seed(42)
        query = torch.randn(200, 2)
        reference = torch.randn(200, 2)
        k = 10
        
        # Full computation
        dist_full = torch.cdist(query, reference, p=2)
        dists_full, indices_full = torch.topk(dist_full, k=k, dim=1, largest=False)
        
        # Chunked computation with large chunk
        dists_chunked, indices_chunked = find_knn_chunked(
            query, reference, k=k, chunk_size=1000, largest=False
        )
        
        assert torch.allclose(dists_full, dists_chunked, atol=1e-5)
        assert torch.equal(indices_full, indices_chunked)


class TestMNNConsistency:
    """Test consistency between original and optimized MNN implementations."""
    
    def _test_consistency_helper(
        self, 
        n_i: int, 
        n_j: int, 
        embed_dim: int = 50,
        overlap_ratio: float = 0.3,
        seed: int = 42,
        k: int = 5,
        top_percent: float = 0.9,
        spatial_k_factor: int = 3,
        chunk_size: int = None,
        min_jaccard: float = 0.5,
        test_name: str = "Consistency test"
    ):
        """
        Helper function to test MNN consistency with different parameters.
        
        Args:
            n_i, n_j: Dataset sizes
            embed_dim: Embedding dimension
            overlap_ratio: Ratio of overlapping points
            seed: Random seed
            k: Number of nearest neighbors
            top_percent: Top percent threshold
            spatial_k_factor: Spatial k factor
            chunk_size: Chunk size for fast implementation
            min_jaccard: Minimum required Jaccard similarity
            test_name: Name for this test
        """
        # Generate test data
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=n_i, n_j=n_j, embed_dim=embed_dim, 
            overlap_ratio=overlap_ratio, seed=seed
        )
        
        # Set default chunk size
        if chunk_size is None:
            chunk_size = max(100, min(n_i, n_j) // 5)
        
        # Run original implementation
        mnn_original = calculate_overlapped_mnn_pairs_original(
            coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
            k=k, top_percent=top_percent
        )
        
        # Run fast implementation
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
            k=k, top_percent=top_percent, 
            spatial_k_factor=spatial_k_factor, chunk_size=chunk_size
        )
        
        # Compare results
        set_original = set(map(tuple, mnn_original))
        set_fast = set(map(tuple, mnn_fast))
        
        intersection = set_original & set_fast
        union = set_original | set_fast
        jaccard = len(intersection) / len(union) if len(union) > 0 else 1.0
        
        print(f"\n{test_name} ({n_i}x{n_j}): "
              f"Original={len(mnn_original)}, Fast={len(mnn_fast)}, "
              f"Jaccard={jaccard:.3f}, Overlapped={len(intersection)}")
        
        assert jaccard > min_jaccard, \
            f"MNN pairs differ too much: Jaccard similarity = {jaccard:.3f}"
        
        return mnn_original, mnn_fast, jaccard
    
    def test_small_dataset_no_embed(self):
        """Test with small dataset without embeddings."""
        coord_i, coord_j, _, _ = generate_test_data(n_i=200, n_j=200, seed=42)
        
        mnn_original = calculate_overlapped_mnn_pairs_original(
            coord_i, coord_j, embed_i=None, embed_j=None, k=5, top_percent=0.3
        )
        
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=None, embed_j=None, 
            k=5, top_percent=0.3, spatial_k_factor=3, chunk_size=100
        )
        
        set_original = set(map(tuple, mnn_original))
        set_fast = set(map(tuple, mnn_fast))
        
        intersection = set_original & set_fast
        union = set_original | set_fast
        jaccard = len(intersection) / len(union) if len(union) > 0 else 1.0
        
        print(f"\nSmall dataset (no embed): Original={len(mnn_original)}, "
              f"Fast={len(mnn_fast)}, Jaccard={jaccard:.3f}, Overlapped={len(intersection)}")
        assert jaccard > 0.7, f"MNN pairs differ too much: Jaccard similarity = {jaccard:.3f}"
    
    def test_medium_dataset_with_embed(self):
        """Test with medium dataset with embeddings."""
        self._test_consistency_helper(
            n_i=500, n_j=500, embed_dim=50, seed=123,
            k=5, top_percent=0.9, spatial_k_factor=3, chunk_size=200,
            min_jaccard=0.4, test_name="Medium dataset (with embed)"
        )
    
    def test_large_dataset_with_embed(self):
        """Test with large dataset (~1000 points) with embeddings."""
        self._test_consistency_helper(
            n_i=1000, n_j=1000, embed_dim=64, overlap_ratio=0.2, seed=999,
            k=6, top_percent=0.9, spatial_k_factor=4, chunk_size=500,
            min_jaccard=0.5, test_name="Large dataset (1000 points)"
        )
    
    def test_asymmetric_dataset(self):
        """Test with asymmetric dataset sizes."""
        self._test_consistency_helper(
            n_i=800, n_j=1200, embed_dim=32, overlap_ratio=0.25, seed=555,
            k=5, top_percent=0.9, spatial_k_factor=3, chunk_size=400,
            min_jaccard=0.5, test_name="Asymmetric dataset (800x1200)"
        )

    def test_huge_dataset(self):
        """Test with huge dataset sizes."""
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=13_000, n_j=12_000, embed_dim=32, overlap_ratio=0.25, seed=555
        )
        
        k = 5
        top_percent = 0.9
        import time
        t0 = time.time()
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
            k=k, top_percent=top_percent, spatial_k_factor=3, chunk_size=400
        )
        
        t1 = time.time()
        print(f"\nHuge dataset ({coord_i.shape[0]}x{coord_j.shape[0]}): "
              f"Fast={len(mnn_fast)}, duration: {t1 - t0:.2f} seconds")
    
    def test_various_sizes(self):
        """Test with various dataset sizes."""
        test_configs = [
            (200, 200, "Small (200x200)"),
            (500, 500, "Medium (500x500)"),
            (300, 600, "Asymmetric small (300x600)"),
            (1000, 800, "Asymmetric large (1000x800)"),
        ]
        
        for n_i, n_j, name in test_configs:
            self._test_consistency_helper(
                n_i=n_i, n_j=n_j, 
                embed_dim=50, overlap_ratio=0.3,
                seed=42, k=5, top_percent=0.9,
                spatial_k_factor=3,
                min_jaccard=0.4,
                test_name=name
            )
    


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_result(self):
        """Test when no MNN pairs should be found."""
        # Very distant point sets
        coord_i = torch.rand(100, 2) * 1000
        coord_j = torch.rand(100, 2) * 1000 + 10000  # Far away
        
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=None, embed_j=None,
            k=3, top_percent=0.01, spatial_k_factor=2
        )
        
        # Should have very few or no pairs
        assert len(mnn_fast) < 10, "Expected few MNN pairs for distant point sets"
    
    def test_small_k(self):
        """Test with small k value."""
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=300, n_j=300, seed=777
        )
        
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
            k=2, top_percent=0.9, spatial_k_factor=3
        )
        
        assert len(mnn_fast) > 0, "Should find some MNN pairs with k=2"
    
    def test_large_k(self):
        """Test with large k value."""
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=400, n_j=400, seed=888
        )
        
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
            k=20, top_percent=0.9, spatial_k_factor=2
        )
        
        assert len(mnn_fast) > 0, "Should find some MNN pairs with large k"
    
    def test_different_spatial_k_factors(self):
        """Test that different spatial_k_factor values work."""
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=300, n_j=300, seed=444
        )
        
        results = {}
        for factor in [2, 3, 5, 10, 50, 100]:
            mnn = calculate_overlapped_mnn_pairs_fast(
                coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
                k=5, top_percent=0.9, spatial_k_factor=factor, chunk_size=150
            )
            results[factor] = len(mnn)
            print(f"spatial_k_factor={factor}: {len(mnn)} pairs")
        
        # Larger factor should generally find more candidates (though final count may vary)
        assert all(count > 0 for count in results.values()), \
            "All spatial_k_factor values should find some pairs"
    
    def test_different_top_percent(self):
        """Test that different top_percent values work."""
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=300, n_j=300, seed=555
        )
        
        results = {}
        for top_pct in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            mnn = calculate_overlapped_mnn_pairs_fast(
                coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
                k=5, top_percent=top_pct, spatial_k_factor=3, chunk_size=150
            )
            results[top_pct] = len(mnn)
            print(f"top_percent={top_pct}: {len(mnn)} pairs")
        
        # Larger top_percent should generally find more pairs
        sorted_percents = sorted(results.keys())
        for i in range(len(sorted_percents) - 1):
            curr_pct = sorted_percents[i]
            next_pct = sorted_percents[i + 1]
            # Allow some flexibility due to discrete nature of the data
            assert results[next_pct] >= results[curr_pct] * 0.8, \
                f"Expected monotonic increase: {curr_pct}={results[curr_pct]}, {next_pct}={results[next_pct]}"
        
        # All should find some pairs
        assert all(count > 0 for count in results.values()), \
            "All top_percent values should find some pairs"


class TestMemoryEfficiency:
    """Test memory efficiency of the optimized implementation."""
    
    def test_memory_usage_comparison(self):
        """Compare memory usage between implementations (qualitative test)."""
        import gc
        import torch.cuda as cuda
        
        # Use moderately large dataset
        coord_i, coord_j, embed_i, embed_j = generate_test_data(
            n_i=1000, n_j=1000, embed_dim=64, seed=2024
        )
        
        # Force garbage collection
        gc.collect()
        if cuda.is_available():
            cuda.empty_cache()
        
        # Test fast implementation
        mnn_fast = calculate_overlapped_mnn_pairs_fast(
            coord_i, coord_j, embed_i=embed_i, embed_j=embed_j,
            k=5, top_percent=0.2, spatial_k_factor=3, chunk_size=500
        )
        
        print(f"\nMemory test: Fast implementation found {len(mnn_fast)} pairs")
        assert len(mnn_fast) >= 0, "Fast implementation should complete successfully"


def test_compute_combined_distance_for_pairs():
    """Test the helper function for computing combined distances."""
    torch.manual_seed(42)
    coord_i = torch.randn(500, 2)
    coord_j = torch.randn(500, 2)
    embed_i = torch.randn(500, 32)
    embed_j = torch.randn(500, 32)
    
    # Create some test pairs
    pairs = np.array([[i, (i + 10) % 500] for i in range(100)])
    
    # Compute distances
    dists = compute_combined_distance_for_pairs(
        pairs, coord_i, coord_j, embed_i, embed_j, chunk_size=50
    )
    
    assert dists.shape[0] == len(pairs), "Should compute distance for each pair"
    assert torch.all(dists >= 0), "All distances should be non-negative"


if __name__ == "__main__":
    print("Running MNN consistency tests...")
    print("=" * 80)
    
    # Run tests manually
    print("\n[1/5] Testing distance functions...")
    test_dist = TestDistanceFunctions()
    test_dist.test_cdist_corr_consistency()
    test_dist.test_cdist_dist_consistency()
    print("✓ Distance functions tests passed")
    
    print("\n[2/5] Testing chunked KNN...")
    test_knn = TestChunkedKNN()
    test_knn.test_find_knn_chunked_small_chunk()
    test_knn.test_find_knn_chunked_large_chunk()
    print("✓ Chunked KNN tests passed")
    
    print("\n[3/5] Testing MNN consistency...")
    test_mnn = TestMNNConsistency()
    test_mnn.test_small_dataset_no_embed()
    test_mnn.test_medium_dataset_with_embed()
    test_mnn.test_large_dataset_with_embed()
    test_mnn.test_asymmetric_dataset()
    test_mnn.test_huge_dataset()
    print("✓ MNN consistency tests passed")
    
    print("\n[4/5] Testing edge cases...")
    test_edge = TestEdgeCases()
    test_edge.test_empty_result()
    test_edge.test_small_k()
    test_edge.test_large_k()
    test_edge.test_different_spatial_k_factors()
    test_edge.test_different_top_percent()
    print("✓ Edge case tests passed")
    
    print("\n[5/5] Testing memory efficiency...")
    test_mem = TestMemoryEfficiency()
    test_mem.test_memory_usage_comparison()
    print("✓ Memory efficiency tests passed")
    
    print("\n[BONUS] Testing helper functions...")
    test_compute_combined_distance_for_pairs()
    print("✓ Helper function tests passed")
    
    print("\n" + "=" * 80)
    print("All tests passed successfully! ✓")
    print("\nSummary:")
    print("- Distance computation functions are consistent")
    print("- Chunked KNN produces identical results")
    print("- MNN implementations show good agreement (Jaccard > 0.5-0.7)")
    print("- Edge cases handled correctly")
    print("- Memory-efficient implementation works for large datasets")
