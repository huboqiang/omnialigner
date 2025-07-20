from typing import Tuple
import dask.array as da
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnialigner.utils.field_transform import tfrs_to_grid_M, calculate_M_from_theta
from omnialigner.utils.point_transform import transfer_landmarks_inv, transform_keypoints, scaled_landmarks_to_raw
from omnialigner.utils.grid_sample import dask_grid_sample
from omnialigner.dtypes import Dask_image_NCHW, Tensor_image_NCHW, Tensor_kpts_N_xy, Tensor_kpts_N_xy_raw, Tensor_tfrs, Tensor_grid_2d

class TFRSModule(nn.Module):
    def __init__(self, 
                 init_tfrs: Tensor_tfrs=torch.FloatTensor([0,0,0,0,0, 1,1]), 
                 tensor_size: Tuple[int, int]=[256, 256], 
                 lambda_L1_angle:float=0.05, 
                 lambda_L1_trans:float=1.0, 
                 lambda_L1_scale:float=10.):
        super().__init__()
        self.params = nn.Parameter(init_tfrs)
        self.tensor_size = tensor_size
        self.lambda_L1_angle = lambda_L1_angle
        self.lambda_L1_trans = lambda_L1_trans
        self.lambda_L1_scale = lambda_L1_scale

    def get_device(self) -> torch.device:
        return self.params.device

    def forward(self, x) -> Tensor_grid_2d:
        params = self.params
        grid_M = tfrs_to_grid_M(params, device=params.device)
        grid = F.affine_grid(grid_M.unsqueeze(0), [1, 1, self.tensor_size[0], self.tensor_size[1]], align_corners=True)
        return grid

    def forward_kpts(self, kpt_M: Tensor_kpts_N_xy) -> Tensor_kpts_N_xy:
        pseudo_image = torch.zeros([1, 1, self.tensor_size[0], self.tensor_size[1]])
        landmark_source_raw = scaled_landmarks_to_raw(kpt_M, pseudo_image[0, 0, :, :])
        M = self._calculate_cv2_affineM()
        transformed_source_points = transform_keypoints(landmark_source_raw, M)
        return transformed_source_points

    def regularization(self) -> torch.Tensor:
        reg_loss = self.lambda_L1_angle * torch.sum(torch.abs(self.params[0:1])) + \
                    self.lambda_L1_trans * torch.sum(torch.abs(self.params[1:3])) + \
                    self.lambda_L1_scale * torch.sum(torch.abs(self.params[3:]))

        return reg_loss
    
    def freeze_layer(self):
        self.params.requires_grad = False


    def unfreeze_layer(self):
        self.params.requires_grad = True

    def _calculate_cv2_affineM(self):
        grid_M = tfrs_to_grid_M(self.params, device=self.params.device)
        tensor_M = torch.eye(3).to(self.params.device).float()
        tensor_M[0:2, :] = grid_M
        M = calculate_M_from_theta(torch.inverse(tensor_M), self.tensor_size[0], self.tensor_size[1])
        return M


def create_initial_grid(rst_module: TFRSModule, **kwargs) -> Tensor_grid_2d:
    grid = rst_module(None)
    return grid

def flip_array(x: Tensor_image_NCHW|Dask_image_NCHW, flip: Tuple[float, float]) -> Tensor_image_NCHW|Dask_image_NCHW:
    axis = [i + 2 for i, flip in enumerate(flip) if flip]
    if isinstance(x, torch.Tensor):
        x = torch.flip(x, axis)
        return x
    elif isinstance(x, Dask_image_NCHW):
        x = da.flip(x, axis=axis)
        return x

    raise TypeError(f"Unsupported data type: {type(x)}")
    
class AffineEngineRawData(nn.Module):
    def __init__(self, 
                 init_tfrs: Tensor_tfrs=torch.FloatTensor([0,0,0,0,0, 1,1]), 
                 params_hw: Tuple[int, int]=(16, 16), 
                 transform_engine: str="torch", 
                 n_workers: int=4):
        super().__init__()
        self.params_hw = params_hw
        self.grid_sample = F.grid_sample
        self.n_workers = n_workers
        self.transform_engine = transform_engine
        if transform_engine == "dask":
            self.grid_sample = dask_grid_sample
            
        self._load_tfrs(init_tfrs=init_tfrs)
        self.grid_up = self.calculate_grid(size=params_hw)
    
    def _load_tfrs(self, init_tfrs: Tensor_tfrs):
        self.tfrs_module = TFRSModule(init_tfrs, tensor_size=self.params_hw)
        height, width = self.params_hw
        self.grid_affine = create_initial_grid(self.tfrs_module)
        
    def _export_rst(self) -> Tensor_tfrs:
        tensor_tfrs = self.tfrs_module.params.detach()
        return tensor_tfrs

    def calculate_grid(self, size: Tuple[int, int]) -> Tensor_grid_2d:
        grid = self.grid_affine
        self.grid_up = F.interpolate(grid.permute(0, 3, 1, 2), size=size, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        return self.grid_up
    
    def show_grid(self, ax=None):
        # grid_up = self.calculate_grid(self.params_hw)
        self.calculate_grid(size=self.params_hw)
        grid_x = self.grid_up[0, :, :, 0].detach().cpu().numpy()
        grid_y = self.grid_up[0, :, :, 1].detach().cpu().numpy()
        
        unseen_idxs = (grid_x>1)+(grid_x<-1)+(grid_y>1)+(grid_y<-1)
        grid_x[unseen_idxs] = np.nan
        grid_y[unseen_idxs] = np.nan
        if ax is None:
            from plotting.matplotlib_init import plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        
        ax.contour(grid_x, levels=50, cmap='bwr')
        ax.contour(grid_y, levels=50, cmap='bwr', linestyles="dashed")
        ax.set_title('Optimized Grid Contours')
        ax.set_xlabel('X Coordinates')
        ax.set_ylabel('Y Coordinates')


    def stn(self, x: Tensor_image_NCHW|Dask_image_NCHW, flip=[0, 0]) -> Tensor_image_NCHW|Dask_image_NCHW:
        x = flip_array(x, flip)
        
        self.calculate_grid(size=x.shape[2:])
        kwargs = {"align_corners": True}
        if self.transform_engine == "dask":
            kwargs["max_workers"] = self.n_workers
            kwargs["verbose"] = "STNetAffine"
        x_transformed = self.grid_sample(x, self.grid_up, **kwargs)
        return x_transformed

    def stn_keypoints(self, kpts: Tensor_kpts_N_xy_raw) -> Tensor_kpts_N_xy_raw:
        kpts_ = transfer_landmarks_inv(kpts.detach(), self.grid_affine)
        return kpts_.to(kpts.device)

    def forward(self, x: Tensor_image_NCHW|Dask_image_NCHW) -> Tensor_image_NCHW|Dask_image_NCHW:
        """
        Apply the STN to the input image or array.
        - dask engine:  
           dask x -> dask, torch x -> dask
        - torch engine:
           dask x -> dask, torch x -> torch
        
        
        Args:
            x (Tensor_image_NCHW|Dask_image_NCHW): The input image or array.
        Returns:
            Tensor_image_NCHW|Dask_image_NCHW: The transformed image or array.
        """
        input_dask = False
        if isinstance(x, Dask_image_NCHW):
            input_dask = True
        
        if self.transform_engine == "dask":
            da_input_NCHW = x
            da_input_NCHW = self.stn(da_input_NCHW)
            return da_input_NCHW
        
        with torch.no_grad():
            if input_dask:
                img_tensor = torch.from_numpy(x.compute()).float()
                tensor = self.stn(img_tensor)
                return da.from_array(tensor.cpu().numpy())
            
            tensor = self.stn(x)
            return tensor
    
    
