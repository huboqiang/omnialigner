from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnialigner.utils.field_transform import tfrs_to_grid_M, calculate_M_from_theta, tfrs_inv
from omnialigner.utils.point_transform import transform_keypoints, scaled_landmarks_to_raw
from omnialigner.dtypes import Grid2DModelDual,Tensor_cv2_affine_M, Tensor_image_NCHW, Tensor_kpts_N_xy, Tensor_trs


class TRSModuleDual(nn.Module, Grid2DModelDual):
    def __init__(self,
                 init_trs: Tensor_trs=None, 
                 tensor_size: Tuple[int, int]=[256, 256], 
                 lambda_L1_angle: float=0.05, 
                 lambda_L1_trans: float=1.0, 
                 lambda_L1_scale: float=10.0, 
                 **kwargs):
        super().__init__()
        if init_trs is None:
            init_trs = torch.zeros(5)
        
        self.tensor_trs = nn.Parameter(init_trs)
        self.init_trs = init_trs.detach().clone()
        self.tensor_size = tensor_size
        self.lambda_L1_angle = lambda_L1_angle
        self.lambda_L1_trans = lambda_L1_trans
        self.lambda_L1_scale = lambda_L1_scale
        self.dev = self.tensor_trs.device if self.tensor_trs is not None else torch.device("cpu")

    def get_device(self):
        return self.tensor_trs.device

    def set_device(self, dev):
        self.dev = dev
        self.tensor_trs.data = self.tensor_trs.data.to(dev) 

    def forward(self, 
                x: Tensor_image_NCHW=None,
                kpts: Tensor_kpts_N_xy=None,
                x_ref: Tensor_image_NCHW=None
        ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:
        params = self.tensor_trs
        return self._forward_image_landmark_trfs(x, kpts, params=params)

    def backforward(self,
                x: Tensor_image_NCHW=None, 
                kpts: Tensor_kpts_N_xy=None, 
                x_ref: Tensor_image_NCHW=None
        ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:
        params = tfrs_inv(self.tensor_trs).to(self.dev)
        return self._forward_image_landmark_trfs(x, kpts, params=params)

    def regularization(self) -> Tuple[torch.FloatTensor, torch.Tensor]:
        """
        return regularization of trs.
        
        Return:
            reg_loss: regularization loss
            reg_weight: regularization loss weight
        """
        self.init_trs = self.init_trs.to(self.dev)
        reg_loss = self.lambda_L1_angle * torch.sum(torch.abs(self.tensor_trs[0:1]-self.init_trs[0:1])) + \
                    self.lambda_L1_trans * torch.sum(torch.abs(self.tensor_trs[1:3]-self.init_trs[1:3])) + \
                    self.lambda_L1_scale * torch.sum(torch.abs(self.tensor_trs[3]-self.tensor_trs[4]))
                    # self.lambda_L1_scale * torch.sum(torch.abs(self.tensor_trs[3:5]-self.init_trs[3:5]))


        return reg_loss, torch.tensor(1.)

    def freeze_layer(self, fwd: bool=True, inv:bool= True):
        if fwd or inv:
            self.tensor_trs.requires_grad = False


    def unfreeze_layer(self, fwd: bool=True, inv:bool= True):
        if fwd or inv:
            self.tensor_trs.requires_grad = True

    def _forward_image_landmark_trfs(self, 
                                     x: Tensor_image_NCHW=None, 
                                     kpts: Tensor_kpts_N_xy=None, 
                                     params: Tensor_trs=torch.zeros(5)) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:
        out_image, out_kpts = None, None
        if x is not None:
            grid_M = tfrs_to_grid_M(params, device=params.device)
            grid = F.affine_grid(grid_M.unsqueeze(0), [1, 1, self.tensor_size[0], self.tensor_size[1]], align_corners=True)
            out_image = F.grid_sample(x, grid=grid, mode="bilinear", align_corners=True)

        if kpts is not None:
            pseudo_image = torch.zeros([1, 1, self.tensor_size[0], self.tensor_size[1]])
            landmark_source_raw = scaled_landmarks_to_raw(kpts, pseudo_image[0, 0, :, :])
            M = self._calculate_cv2_affineM(params)
            out_kpts = transform_keypoints(landmark_source_raw, M)
            size_ = torch.tensor(self.tensor_size).to(self.dev)
            out_kpts = out_kpts / size_
        
        return out_image, out_kpts

    def _calculate_cv2_affineM(self, params: Tensor_trs) -> Tensor_cv2_affine_M:
        grid_M = tfrs_to_grid_M(params, device=params.device)
        tensor_M = torch.eye(3).to(params.device).float()
        tensor_M[0:2, :] = grid_M
        
        M = calculate_M_from_theta(torch.inverse(tensor_M), self.tensor_size[0], self.tensor_size[1])
        return M
