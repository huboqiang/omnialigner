
import unittest
import torch
from omnialigner.dtypes import Tensor_tfrs
from omnialigner.utils.field_transform import tfrs_to_grid_M, grid_M_to_tfrs, tfrs_inv

class PseudoTFRS(torch.nn.Module):
    def __init__(self, tensor_tfrs: Tensor_tfrs, use_easy_model=True):
        super().__init__()
        self.tensor_tfrs = torch.nn.Parameter(tensor_tfrs)
        self.zeros = torch.zeros(1)[0]
        self.ones = torch.ones(1)[0]
        self.use_easy_model = use_easy_model

    def easy_forward_model(self):
        _, tx, ty, _, _, fx, fy = self.tensor_tfrs
        TF = torch.stack([
            torch.stack([fx, self.zeros, tx], dim=-1),
            torch.stack([self.ones, fy, ty], dim=-1),
        ], dim=0)
        return TF

    def forward(self):
        if self.use_easy_model:
            return self.easy_forward_model()
        
        return tfrs_to_grid_M(self.tensor_tfrs)

class TestFieldTransform(unittest.TestCase):
    def pseudo_TRFS(self, model: torch.nn.Module):
        image = torch.randn(1, 1, 5, 5)
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(10):
            optim.zero_grad()
            TF_grid = model.forward()
            grid_2d = torch.nn.functional.affine_grid(TF_grid.unsqueeze(0), size=(1, 1, 5, 5), align_corners=False)
            transformed_image = torch.nn.functional.grid_sample(image, grid_2d, align_corners=False)
            loss = transformed_image.sum()
            loss.backward()
            optim.step()

        return TF_grid

    def test_if_grad_keeping_in_const(self):
        model = PseudoTFRS(torch.ones(7))
        grid_M = self.pseudo_TRFS(model)

        assert(grid_M[0, 0] != 1)
        assert(grid_M[0, 1] == 0)
        assert(grid_M[0, 2] != 0)
        assert(grid_M[1, 0] == 1)
        assert(grid_M[1, 1] != 1)
        assert(grid_M[1, 2] != 1)

    def test_if_grad_keeping_in_tfrs(self):
        tfrs = torch.nn.Parameter(torch.Tensor([0, 0, 0, 0, 0, 1, 1]))
        grid_M_init = tfrs_to_grid_M(tfrs)
        model = PseudoTFRS(tfrs, use_easy_model=False)
        grid_M = self.pseudo_TRFS(model)

        assert(torch.abs(grid_M - grid_M_init).sum() > 1e-2)


    def test_if_trfs_inverse_is_correc_1(self):
        tensor_tfrs = torch.nn.Parameter(torch.FloatTensor([0.1, 0.2, 0.3, 0.01, -0.01, 1, 1]))

        image = torch.randn(1, 1, 5, 5)
        
        

        grid_M = tfrs_to_grid_M(tensor_tfrs)
        grid_2d = torch.nn.functional.affine_grid(grid_M.unsqueeze(0), size=(1, 1, 5, 5), align_corners=False)
        _ = torch.nn.functional.grid_sample(image, grid_2d, align_corners=False)

        tensor_tfrs_inv = grid_M_to_tfrs(tfrs_inv(tensor_tfrs)[0:2])
        # print(tensor_tfrs, tensor_tfrs_inv)
        
        tensor_tfrs_inv_inv = grid_M_to_tfrs(tfrs_inv(tensor_tfrs_inv)[0:2])
        assert(torch.abs(tensor_tfrs - tensor_tfrs_inv_inv).mean() < 1e-2)
        
    def test_if_trfs_inverse_is_correc_2(self):
        l_tests = [
            [-1.7204,  0.1675, -0.2034, -0.2188,  0.0605, -1.0000,  1.0000],
            [-0.0464,  0.4517, -0.5064, -0.3462,  0.1886,  1.0000,  1.0000]
        ]
        for tfrs in l_tests:
            tensor_tfrs = torch.FloatTensor(tfrs)
            grid_M = tfrs_to_grid_M(tensor_tfrs)
            tensor_tfrs_ = grid_M_to_tfrs(
                grid_M
            )
            # print(tensor_tfrs, tensor_tfrs_)
            assert(torch.abs(tensor_tfrs - tensor_tfrs_).mean() < 1e-2)
            # print(grid_M, "R1")
            grid_M_ = tfrs_to_grid_M(tensor_tfrs_)
            assert(torch.abs(grid_M - grid_M_).mean().item() < 1e-2)


if __name__ == "__main__":
    unittest.main()