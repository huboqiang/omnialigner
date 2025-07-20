import importlib
import pdb
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

from omnialigner.dtypes import Grid2DModelDual, Tensor_image_NCHW, Tensor_kpts_N_xy, Tensor_trs
from omnialigner.utils.field_transform import create_gaussian_kernel, generate_grid, warp_tensor, resample_displacement_field_to_size, tfrs_to_grid_M
from omnialigner.utils.point_transform import warp_landmark_grid_faiss as warp_landmark_grid

def cubic_bspline_basis(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0,1)
    b_m2 = ((1-u)**3)/6
    b_m1 = (3*u**3 - 6*u**2 + 4)/6
    b0   = (-3*u**3 + 3*u**2 + 3*u + 1)/6
    b_p1 = u**3/6
    return torch.stack((b_m2,b_m1,b0,b_p1), dim=-1)

def fill_nan(parameter, value=0.0):
    with torch.no_grad():
        parameter.data[torch.isnan(parameter.data)] = value


class BSplineFFD2D(nn.Module):
    def __init__(self, img_hw: int, cp_spacing: Tuple[int, int]=(16, 16)):
        super().__init__()
        self.H, self.W = img_hw
        sx = int(self.W/cp_spacing[0])+1
        sy = int(self.H/cp_spacing[1])+1
        self.gx, self.gy = cp_spacing
        self.omega = nn.Parameter(torch.zeros(sy, sx, 2))  # (sy,sx,2)

    def forward(self, grid: torch.Tensor):                                # grid (N,2)
        x, y = grid[:,0]*self.W/2 + self.W/2, grid[:,1]*self.H/2 + self.H/2
        ix, iy = (x/self.gx).floor().long(), (y/self.gy).floor().long()
        u, v = (x-ix*self.gx)/self.gx, (y-iy*self.gy)/self.gy
        Bu, Bv = cubic_bspline_basis(u), cubic_bspline_basis(v)  # (N,4)
        Buv = Bu.unsqueeze(2)*Bv.unsqueeze(1)                    # (N,4,4)

        pad = 2
        w = F.pad(self.omega, (0,0,pad,pad,pad,pad))             # (sy+4,sx+4,2)
        off = torch.tensor([-2,-1,0,1], device=grid.device)

        ix = (ix+pad)[:,None] + off                              # (N,4)
        iy = (iy+pad)[:,None] + off
        w_nb = w[iy.unsqueeze(2), ix.unsqueeze(1)]               # (N,4,4,2)
        disp = (Buv.unsqueeze(-1)*w_nb).sum((1,2))               # (N,2)
        disp_norm = torch.stack([disp[:,0]*2/self.W,
                                 disp[:,1]*2/self.H], dim=-1)
        return disp_norm

class DeformKANGrid2D(nn.Module, Grid2DModelDual):
    def __init__(self,
                 init_trs:Tensor_trs=None,
                initial_displacement_field=None,
                current_epoch=0,
                 tensor_size=(256,256),
                 cp_spacing=(16,16),
                 kan_width=(2,32,2),
                 kan_grid=9,
                 kan_k=3,
                 disp_type="bspline",
                 kernel_size=5,
                 sigma=1.0,
                 alpha=1.0,
                 regularization_function:Dict[str,dict]={},
                **kwargs
        ):
        super().__init__()
        self.tensor_size = tensor_size
        self.regularization_function = regularization_function
        H, W = tensor_size
        self.disp_type = disp_type
        self.bspline = BSplineFFD2D((H,W), cp_spacing)
        self.kan = KAN(
            width=list(kan_width),
            k=kan_k,
            grid=kan_grid,
            base_fun=lambda u: cubic_bspline_basis(u).sum(-1),
            affine_trainable=False,
            sp_trainable=False,
            sb_trainable=False
        )
        self.kernel = create_gaussian_kernel(kernel_size, sigma)
        if init_trs is None:
            init_trs = torch.zeros(5)

        self.tensor_trs = init_trs
        self._cache_disp = None
        self.alpha = alpha

    # -------------------- disp property --------------------
    @property
    def disp(self):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1,1,self.tensor_size[0], device=self.bspline.omega.device),
            torch.linspace(-1,1,self.tensor_size[1], device=self.bspline.omega.device),
            indexing='ij')
        pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        with torch.no_grad():
            coarse = self.bspline(pts)                    # (H*W,2)
            fine   = self.kan(pts)                        # (H*W,2)
        disp = (coarse + fine).reshape(
            1, self.tensor_size[0], self.tensor_size[1], 2
        )#.permute(0,3,1,2)
        return disp                                       # (1,2,H,W)


    # -------------------- Grid2DModelDual API --------------------
    def forward(self,
                x: Tensor_image_NCHW=None,
                kpts: Tensor_kpts_N_xy=None,
                x_ref: Tensor_image_NCHW=None
        ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:
        return self._forward_image_landmark_trs(x, kpts, x_ref)

    def _forward_image_landmark_trs(self,
            x: Tensor_image_NCHW=None,
            kpts: Tensor_kpts_N_xy=None,
            x_ref: Tensor_image_NCHW=None
        ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:

        grid_M = tfrs_to_grid_M(self.tensor_trs, device=self.dev)
        h, w = self.tensor_size
        if x is not None:
            h, w = x.shape[2], x.shape[3]

        grid = F.affine_grid(grid_M.unsqueeze(0), [1, 1, h, w], align_corners=True)   
        warped_source = None

        displacement_field_smoothed = self.disp
        if x is not None:
            print(x.shape, displacement_field_smoothed.shape)
            warped_source = warp_tensor(x, displacement_field_smoothed, grid=grid, device=self.dev)

        kpts_new_scaled = None
        if kpts is not None:
            disp = displacement_field_smoothed
            resized_ = (disp.shape[1], disp.shape[2])
            grid_resized = resample_displacement_field_to_size(grid, resized_)
            final_hw = (100, 100)
            if resized_[0] > 50:
                final_hw = (500, 500)
            sampling_grid = resample_displacement_field_to_size(grid_resized + disp, final_hw)
            kpts_new_scaled = warp_landmark_grid(kpts, grid=sampling_grid)
            kpts_new_scaled = kpts_new_scaled[:, 0, :]

        return warped_source, kpts_new_scaled

    def backforward(self, x, kpts, x_ref=None):
        return None, None

    # -------------------- device helper --------------------
    def set_device(self, dev:torch.device):
        self.dev = dev
        self.bspline.to(dev)
        self.kan.to(dev)
        self.kernel = self.kernel.to(dev)
        self.tensor_trs = self.tensor_trs.to(dev)

    def get_device(self):  # noqa
        return next(self.parameters()).device

    # -------------------- reg loss --------------------
    def regularization(self):
        total_reg_loss = 0.0
        for func_name, element in self.regularization_function.items():
            weight = element.get("weight", 1.0)
            params = element.get("params", {})

            module_name, func_name = func_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            reg = func(self.disp, **params)
            if func_name == "diffusion_relative":
                reg = self.alpha*reg

            total_reg_loss += weight * reg

        return total_reg_loss, torch.tensor(1.)

    def freeze_layer(self, fwd:bool=True, inv:bool=True):
        if fwd or inv:
            for p in self.bspline.parameters():
                p.requires_grad=False

            for p in self.kan.parameters():
                p.requires_grad=False

    def unfreeze_layer(self, fwd:bool=True, inv:bool=True):
        if fwd or inv:
            for p in self.bspline.parameters():
                p.requires_grad=True

            for p in self.kan.parameters():
                p.requires_grad=True
