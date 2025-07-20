import importlib
import pdb
from typing import Tuple, Dict
import torch

from omnialigner.align.models.grid_2d.bsplines import b_splines, create_control_points
from omnialigner.align.models.loss import *
from omnialigner.dtypes import Grid2DModelDual, Tensor_disp_2d, Tensor_grid_2d, Tensor_image_NCHW, Tensor_trs, Tensor_kpts_N_xy, Np_image_HWC
from omnialigner.utils.field_transform import tfrs_to_grid_M, create_gaussian_kernel, smooth_displacement_with_kernel, resample_displacement_field_to_size, generate_grid
from omnialigner.utils.point_transform import warp_landmark_grid_faiss as warp_landmark_grid
from omnialigner.utils.image_viz import im2tensor

torch.use_deterministic_algorithms(False)

def fill_nan(parameter, value=0.0):
    with torch.no_grad():
        parameter.data[torch.isnan(parameter.data)] = value


class DeeperHistRegModule(nn.Module, Grid2DModelDual):
    def __init__(self,
            init_trs:Tensor_trs=None,
            initial_displacement_field=None,
            current_epoch=0,
            tensor_size=[256, 256],
            disp_type="nonrigid",
            regularization_function={},
            alphas=[ 1.5, 1.5, 1.5, 1.5, 1.5, 1.8, 2.1],
            kernel_size=5,
            sigma=1.0,
            cp_spacing=(1.0, 1.0),
            splines_type="cubic",
            **kwargs
        ):
        super().__init__()
        if init_trs is None:
            init_trs = torch.zeros(5)
        
        self.tensor_trs = init_trs
        self.init_trs = init_trs.detach().clone()
        self.alpha = alphas[current_epoch]
        pseudo_image = torch.zeros([1, 3, tensor_size[0], tensor_size[1]])
        self.disp_type = disp_type
        self.splines_type = splines_type
        self.cp_spacing = cp_spacing

        if initial_displacement_field is None:
            displacement_field = self.__init_displacement_field(pseudo_image)
        else:
            displacement_field = resample_displacement_field_to_size(initial_displacement_field, (pseudo_image.size(2), pseudo_image.size(3))).detach().clone()

        self.kernel = create_gaussian_kernel(kernel_size, sigma)
        self.displacement_field = nn.Parameter(displacement_field)
        if getattr(self, 'disp_type') == "bspline" and initial_displacement_field is not None:
            self.displacement_field = initial_displacement_field

        self.tensor_size = tensor_size
        self.regularization_function = regularization_function


    def __init_displacement_field(self, pseudo_image):
        if getattr(self, 'disp_type') == "bspline":
            return create_control_points(pseudo_image, spacing=self.cp_spacing, splines_type=self.splines_type)

        displacement_field = create_identity_displacement_field(pseudo_image).detach().clone()
        return displacement_field

    @property
    def disp(self):
        """Dynamic return smoothed disp."""
        if getattr(self, 'disp_type') == "bspline":
            out_disp = b_splines(
                    self.displacement_field.permute(0, 3, 1, 2),
                    self.displacement_field,
                    self.cp_spacing,
                    splines_type=self.splines_type,
                    normalize=True
            )
            fill_nan(out_disp)
            return out_disp

            # if self.displacement_field.requires_grad:
            #     return b_splines(
            #         self.displacement_field,
            #         self.displacement_field,
            #         self.cp_spacing,
            #         splines_type=self.splines_type,
            #         normalize=True
            #     )
        
        fill_nan(self.displacement_field)
        return smooth_displacement_with_kernel(self.displacement_field, self.kernel)

    @disp.setter
    def disp(self, value):
        """Support re-assignment to disp."""
        with torch.no_grad():
            self.displacement_field.data = value

    def get_device(self) -> torch.device:
        return self.displacement_field.device


    def set_device(self, dev: torch.device):
        self.dev = dev
        self.displacement_field = nn.Parameter(self.displacement_field.to(dev))
        if self.tensor_trs is not None:
            self.tensor_trs = self.tensor_trs.to(self.dev)

        if self.kernel is not None:
            self.kernel = self.kernel.to(self.dev)

    def forward(self, 
            x: Tensor_image_NCHW=None, 
            kpts: Tensor_kpts_N_xy=None, 
            x_ref: Tensor_image_NCHW=None
        ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:
        return self._forward_image_landmark_trs(x, kpts, x_ref)

    def forward_raw_size(self, x: Tensor_image_NCHW=None):
        disp = resample_displacement_field_to_size(self.disp, (x.size(2), x.size(3)))
        grid = generate_grid(x).to(disp.device)
        sampling_grid = grid + disp
        return F.grid_sample(x.to(disp.device), sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)


    def backforward(self, 
            x: Tensor_image_NCHW=None, 
            kpts: Tensor_kpts_N_xy=None, 
            x_ref: Tensor_image_NCHW=None
        ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy]:
        return None, None


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

            if torch.isnan(reg):
                pdb.set_trace()
            
            total_reg_loss += weight * reg

        return total_reg_loss, torch.tensor(1.)

    def freeze_layer(self, fwd: bool=True, inv:bool= True):
        if fwd or inv:
            self.displacement_field.requires_grad = False


    def unfreeze_layer(self, fwd: bool=True, inv:bool= True):
        if fwd or inv:
            self.displacement_field.requires_grad = True

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
        channels = self.displacement_field.shape[0]
        displacement_field_smoothed = self.disp
        if x is not None:
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



def warp_tensor(
        tensor: Tensor_image_NCHW, 
        displacement_field: Tensor_disp_2d, 
        grid: Tensor_grid_2d=None, 
        mode: str='bilinear', 
        padding_mode: str='zeros', 
        align_corners: bool=False,
        device: torch.device=None
    ) -> torch.Tensor:
    """
    Transforms a tensor with a given displacement field.
    Uses F.grid_sample for the structured data interpolation (only linear and nearest supported).
    Be careful - the autogradient calculation is possible only with mode set to "bilinear".

    Args:
    ----------
    tensor : torch.Tensor
        The tensor to be transformed (BxYxXxZxD)
    displacement_field : torch.Tensor
        The PyTorch displacement field (BxYxXxZxD)
    grid : torch.Tensor (optional)
        The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    device : str
        The device to generate the warping grid if not provided

    Returns
    ----------
    transformed_tensor : torch.Tensor
        The transformed tensor (BxYxXxZxD)
    """
    displacement_field = resample_displacement_field_to_size(displacement_field, (tensor.size(2), tensor.size(3)))
    if grid is None:
        grid = generate_grid(tensor=displacement_field, device=device)
    
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return transformed_tensor


def create_identity_displacement_field(tensor : torch.Tensor) -> torch.Tensor:
    """
    TODO
    """
    return torch.zeros((tensor.size(0), tensor.size(2), tensor.size(3)) + (2,)).type_as(tensor)


def image_aligner_DeeperHistReg(
        image_F: Np_image_HWC, 
        image_moved: Np_image_HWC, 
        nonrigid_registration_params: Dict[str, str]=None,
        init_displacement_field: torch.Tensor=None,
        device: torch.device = torch.device("cpu")
    ):
    
    from deeperhistreg.dhr_registration.dhr_nonrigid_registration.io_nonrigid import instance_optimization_nonrigid_registration # type: ignore

    tensor_F = im2tensor(image_F)[:, 0:1, :, :].to(device)

    tensor_moved = im2tensor(image_moved)[:, 0:1, :, :].to(device)
    
    if init_displacement_field is None:
        H, W = tensor_F.shape[2], tensor_F.shape[3]
        deformation_field = F.affine_grid(torch.Tensor([[[1,0,0], [0,1,0]]]), size=[1,1,H,W]).to(device)
        size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
        grid = generate_grid(tensor_size=size, device=deformation_field.device)
        init_displacement_field = deformation_field - grid

    if nonrigid_registration_params is None:
        nonrigid_registration_params = {
            'save_results': True,
            'nonrigid_registration_function': 'instance_optimization_nonrigid_registration',
            'device': str(device),
            'echo': False,
            'cost_function': 'ncc_local',
            'cost_function_params': {'win_size': 7},
            'regularization_function': 'diffusion_relative',
            'regularization_function_params': {},
            'registration_size': 8192,
            'num_levels': 9,
            'used_levels': 9, 
            'iterations': [100, 100, 100, 100, 100, 100, 100, 200, 200],
            'learning_rates': [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0015, 0.001],
            'alphas': [0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.054, 0.063]
        }
        
    displacement_field = instance_optimization_nonrigid_registration(tensor_moved, tensor_F, init_displacement_field, nonrigid_registration_params)
    res = warp_tensor(im2tensor(image_moved), displacement_field.cpu() )
    return res, displacement_field



def image_aligner_DeeperHistReg_bsplines(image_F: Np_image_HWC, image_moved: Np_image_HWC, nonrigid_registration_params=None):
    device = torch.device("cpu")
    from deeperhistreg.dhr_registration.dhr_nonrigid_registration.io_bsplines import instance_optimization_bsplines_registration # type: ignore

    tensor_F = im2tensor(image_F)[:, 0:1, :, :].to(torch.device("cuda:0"))

    tensor_moved = im2tensor(image_moved)[:, 0:1, :, :].to(torch.device("cuda:0"))
    H, W = tensor_F.shape[2], tensor_F.shape[3]

    deformation_field = F.affine_grid(torch.Tensor([[[1,0,0], [0,1,0]]]), size=[1,1,H,W]).to(torch.device("cuda:0"))
    size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
    grid = generate_grid(tensor_size=size, device=deformation_field.device)
    init_displacement_field = deformation_field - grid

    if nonrigid_registration_params is None:
        nonrigid_registration_params = {
            'save_results': True,
            'nonrigid_registration_function': 'instance_optimization_bsplines_registration',
            'device': 'cuda:0',
            'echo': False,
            'cost_function': 'ncc_local',
            'cost_function_params': {'win_size': 7},
            'regularization_function': 'diffusion_relative',
            'regularization_function_params': {},
            'registration_size': 8192,
            'num_levels': 7,
            'used_levels': 7, 
            'iterations': [100, 100, 100, 100, 100, 200, 200],
            'learning_rates': [0.005, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025],
            'alphas': [0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045],
            'cp_spacing': (0.1, 0.1),       # spacing between control points
            'splines_type': 'cubic', # [ 'cubic', 'linear']
        }
        
    grid = generate_grid(tensor_moved)
    displacement_field = instance_optimization_bsplines_registration(tensor_moved, tensor_F, init_displacement_field, nonrigid_registration_params)
    res = warp_tensor(im2tensor(image_moved), displacement_field.cpu() )
    return res, displacement_field