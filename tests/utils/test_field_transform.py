import unittest

import numpy as np
import torch
import torch.nn.functional as F
from omnialigner.utils.field_transform import grid_to_disp, disp_to_grid, grid_2d_to_disp_field, disp_field_to_grid_2d


class TestFieldTransform(unittest.TestCase):
    def test_grid_to_disp_and_disp_to_grid(self):
        # Create a sample grid
        N, H, W = 1, 4, 4
        theta = torch.eye(2, 3).unsqueeze(0)  # Identity transformation

        # Generate the grid using F.affine_grid
        grid = F.affine_grid(theta, [N, 1, H, W])


        # Convert grid to displacement field
        disp = grid_to_disp(grid)

        # Convert displacement field back to grid
        grid_reconstructed = disp_to_grid(disp)

        # Check if the reconstructed grid is close to the original grid
        self.assertTrue(np.allclose(grid.numpy(), grid_reconstructed.numpy(), atol=1e-6), "Test failed: The reconstructed grid does not match the original grid.")

        print("Test passed: The reconstructed grid matches the original grid.")

    def test_grid_2d_to_disp_field_and_disp_field_to_grid_2d(self):
        # Create a sample grid
        N, H, W = 1, 4, 4
        theta = torch.eye(2, 3).unsqueeze(0)  # Identity transformation

        # Generate the grid using F.affine_grid
        grid = F.affine_grid(theta, [N, 1, H, W])

        # Convert grid to displacement field
        disp_field = grid_2d_to_disp_field(grid)

        # Convert displacement field back to grid
        grid_reconstructed = disp_field_to_grid_2d(disp_field)

        # Check if the reconstructed grid is close to the original grid
        self.assertTrue(np.allclose(grid.numpy(), grid_reconstructed.numpy(), atol=1e-6), "Test failed: The reconstructed grid does not match the original grid.")

    def test_disp_field_to_grid_2d_and_grid_2d_to_disp_field(self):
        # Create a sample displacement field
        N, H, W = 1, 4, 4
        disp_field = torch.zeros((N, H, W, 2), dtype=torch.float32)

        # Set some arbitrary displacements
        disp_field[0, :, :, 0] = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                                [0.5, 0.6, 0.7, 0.8],
                                                [0.9, 1.0, 1.1, 1.2],
                                                [1.3, 1.4, 1.5, 1.6]])
        disp_field[0, :, :, 1] = torch.tensor([[0.2, 0.3, 0.4, 0.5],
                                                [0.6, 0.7, 0.8, 0.9],
                                                [1.0, 1.1, 1.2, 1.3],
                                                [1.4, 1.5, 1.6, 1.7]])

        # Convert displacement field to grid
        grid = disp_field_to_grid_2d(disp_field)

        # Convert grid back to displacement field
        disp_field_reconstructed = grid_2d_to_disp_field(grid)

        # Check if the reconstructed displacement field is close to the original displacement field
        self.assertTrue(np.allclose(disp_field.numpy(), disp_field_reconstructed.numpy(), atol=1e-6), "Test failed: The reconstructed displacement field does not match the original displacement field.")

if __name__ == "__main__":
    unittest.main()
    print("All tests passed.")