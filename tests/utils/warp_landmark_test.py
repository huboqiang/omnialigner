import unittest
import torch
import time
import torch.nn.functional as F
from ...utils.point_transform import warp_landmark_grid, warp_landmark_grid_faiss

class TestWarpLandmarkGrid(unittest.TestCase):
    def test_warp_landmark_grid_functions(self):
        # Generate sample data
        H, W = 1000, 1000
        N = 10000
        k = 1

        torch.manual_seed(42)
        landmarks = torch.rand(N, 2)

        theta = torch.randn(1, 2, 3)
        theta[:, :, 2] = 0
        theta[:, :, :2] *= 0.1
        grid = F.affine_grid(theta, size=(1, 1, H, W), align_corners=False)


        # Run original function
        start_time = time.time()
        result_original = warp_landmark_grid(landmarks, grid, k)[:, 0, :]
        original_time = time.time() - start_time
        print(f"Original function time: {original_time:.4f} seconds")

        # Run Faiss function
        start_time = time.time()
        result_faiss = warp_landmark_grid_faiss(landmarks, grid, k)[:, 0, :]
        faiss_time = time.time() - start_time
        print(f"Faiss function time: {faiss_time:.4f} seconds")

        # Check results
        diff = torch.abs(result_original - result_faiss)
        max_diff = torch.max(diff)
        print(f"Maximum difference between results: {max_diff:.6f}, avg: {torch.mean(diff):.6f}")

        is_close = torch.allclose(result_original, result_faiss, atol=1e-2)
        self.assertTrue(is_close)

        # Calculate speedup
        speedup = original_time / faiss_time
        print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    unittest.main()