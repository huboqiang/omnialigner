import torch
import torch.nn.functional as F
import dask.array as da
from omnialigner.utils.grid_sample import dask_grid_sample
import unittest



class TestChunkedGridSample(unittest.TestCase):
    def setUp(self):
        # Define different test cases as tuples of (N, C, H, W, H_out, W_out, N_chunk_size, C_chunk_size)
        self.test_cases = [
            (4, 32, 128, 128, 128, 128, 1, 8),
            (8, 64, 256, 256, 256, 256, 2, 16),
            (8, 64, 512, 512, 256, 256, 2, 16),
            (2, 16, 2560+250, 2560, 256, 256, 2, 16),
            # (16, 8, 5120, 5120, 1024, 1024, 2, 8),
            # (2, 16, 51200, 51200, 1024, 1024, 2, 16),
        ]

    def create_random_grid(self, N, H_out, W_out):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_out),
            torch.linspace(-1, 1, W_out),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)
        return grid + 0.1 * torch.randn_like(grid)

    def test_chunked_grid_sample(self):
        
        for N, C, H, W, H_out, W_out, N_chunk_size, C_chunk_size in self.test_cases:
            with self.subTest(N=N, C=C, H=H, W=W, H_out=H_out, W_out=W_out):
                input_tensor = torch.randn(N, C, H, W)
                grid = self.create_random_grid(N, H_out, W_out)

                output_direct = F.grid_sample(input_tensor, grid, align_corners=True)
                
                da_array = da.from_array(input_tensor.cpu().numpy())
                print("Testing dask_grid_sample")        
                output_dask = dask_grid_sample(da_array, grid, align_corners=True)
                output_tensor = torch.from_numpy(output_dask.compute())
                self.assertTrue(torch.allclose(output_direct, output_tensor, atol=1e-6),
                                f"Outputs do not match for output_tensor with max diff: {(output_direct - output_tensor).abs().max().item()}")
                
                

if __name__ == '__main__':
    unittest.main()