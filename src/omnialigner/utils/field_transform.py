import torch
from typing import Tuple
import torch.nn.functional as F

from omnialigner.dtypes import Tensor_disp_2d, Tensor_grid_M, Tensor_grid_2d, Tensor_trs, Tensor_tfrs, Np_disp_2d, Tensor_cv2_affine_M, Tensor_image_NCHW

def grid_to_disp(grid: Tensor_grid_2d) -> Np_disp_2d:
    """Converts a sampling grid to a displacement field.

    Args:
        grid: A tensor of shape (N, H_out, W_out, 2) representing the sampling grid.
            Compatible with torch.nn.functional.grid_sample format.

    Returns:
        A numpy array of shape (H, W, N, 2) representing the displacement field.
        Compatible with scipy.ndimage.map_coordinates format.
    """
    N, H, W, _ = grid.shape

    grid = grid.permute(0, 2, 1, 3).flip(-1)
    disp_field = grid - F.affine_grid(torch.eye(3, device=grid.device)[:2, :].unsqueeze(0), (1, 1, H, W))
    disp_field = ((disp_field / 2) * (torch.tensor([H, W], device=grid.device) - 1)).flip(-1)
    disp_field = disp_field.permute(1, 2, 0, 3).detach().cpu().numpy()

    return disp_field

def disp_to_grid(disp: Np_disp_2d) -> Tensor_grid_2d:
    """
    Convert a displacement field to a sampling grid.

    Args:
        disp: numpy array [H, W, 1, 2] 
            Compatible with scipy.ndimage.map_coordinates format.
    
    Returns:
        grid: torch.Tensor [1, H, W, 2]
            Compatible with torch.nn.functional.grid_sample format.
    """
    H, W, N, _ = disp.shape
    
    disp_tensor = torch.from_numpy(disp).float().permute(2, 0, 1, 3)    
    disp_tensor = disp_tensor.flip(-1)
    disp_tensor = disp_tensor / (torch.tensor([H, W]) - 1) * 2
    disp_tensor = disp_tensor.flip(-1)
    
    identity_grid = F.affine_grid(
        torch.eye(3)[:2, :].unsqueeze(0), 
        (1, 1, H, W), 
    )
    disp_tensor = disp_tensor.permute(0, 2, 1, 3)
    grid = identity_grid + disp_tensor
    return grid


def generate_grid(tensor: Tensor_grid_2d|Tensor_disp_2d=None, tensor_size: torch.Tensor=None, device: torch.device=None) -> Tensor_grid_2d:
    """Generates an identity sampling grid for a given tensor size.

    This function creates a regular grid that can be used as a base for image transformations.
    The grid is generated in relative coordinates for use with warp_tensor (align_corners=False).

    Args:
        tensor: Optional tensor to be used as a template for grid generation.
        tensor_size: Optional tensor size or torch.Size to specify the dimensions of the grid.
        device: Optional torch device to place the generated grid on.

    Returns:
        A tensor of shape (N, H, W, 2) representing the identity sampling grid.
    """
    if tensor is not None:
        tensor_size = tensor.permute(0, 3, 1, 2).size()
    if device is None:
        identity_transform = torch.eye(len(tensor_size)-1)[:-1, :].unsqueeze(0).type_as(tensor)
    else:
        identity_transform = torch.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    
    identity_transform = torch.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

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
        tensor : torch.Tensor                   The tensor to be transformed (BxYxXxZxD)
        displacement_field : torch.Tensor       The PyTorch displacement field (BxYxXxZxD)
        grid : torch.Tensor (optional)          The input identity grid (optional - may be provided to speed-up the calculation for iterative algorithms)
        mode : str                              The interpolation mode ("bilinear" or "nearest")
        device : str                            The device to generate the warping grid if not provided

    Return:
        transformed_tensor : torch.Tensor       The transformed tensor (BxYxXxZxD)
    """
    displacement_field = resample_displacement_field_to_size(displacement_field, (tensor.size(2), tensor.size(3)))
    if grid is None:
        grid = generate_grid(tensor=displacement_field, device=device)
    
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return transformed_tensor


def disp_field_to_grid_2d(disp: Tensor_disp_2d, grid: Tensor_grid_2d=None) -> Tensor_grid_2d:
    """Converts a PyTorch displacement field to a sampling grid.

    This function transforms a displacement field into a format suitable for image sampling,
    optionally using a provided base grid.

    Args:
        disp: A tensor of shape (N, H, W, 2) representing the displacement field.
        grid: Optional tensor of shape (N, H, W, 2) representing the base sampling grid.
            If None, an identity grid will be generated.

    Returns:
        A tensor of shape (N, H, W, 2) representing the transformed sampling grid.
    """
    if grid is None:
        grid = generate_grid(tensor=disp, device=disp.device)

    displacement_field = resample_displacement_field_to_size(disp, (grid.size(1), grid.size(2)))
    sampling_grid = grid + displacement_field
    return sampling_grid


def grid_2d_to_disp_field(grid: Tensor_grid_2d=None, grid_affine: Tensor_grid_2d=None) -> Tensor_disp_2d:
    """Converts a PyTorch displacement field to a sampling grid.

    This function transforms a displacement field into a format suitable for image sampling,
    optionally using a provided base grid.

    Args:
        disp: A tensor of shape (N, H, W, 2) representing the displacement field.
        grid: Optional tensor of shape (N, H, W, 2) representing the base sampling grid.
            If None, an identity grid will be generated.

    Returns:
        A tensor of shape (N, H, W, 2) representing the transformed sampling grid.
    """
    if grid_affine is None:
        grid_affine = generate_grid(tensor=grid, device=grid.device)

    displacement_field = resample_displacement_field_to_size(grid, (grid_affine.size(1), grid_affine.size(2)))
    sampling_grid = displacement_field - grid_affine
    return sampling_grid



def calculate_theta_from_M(meta: Tensor_cv2_affine_M, h: int, w: int, device: torch.device=None) -> Tensor_grid_M:
    """Converts OpenCV-style affine matrix to PyTorch theta parameters.

    Args:
        meta: OpenCV-style affine transformation matrix.
        h: Height of the image.
        w: Width of the image.
        device: Optional torch device to place the output tensor on.

    Returns:
        A tensor representing the theta parameters compatible with torch.nn.functional.affine_grid.
    """
    if device is None:
        device = meta.device
    theta = torch.zeros_like(meta).to(device)
    theta[0,0] = meta[0,0]
    theta[0,1] = meta[0,1] * h / w
    theta[0,2] = meta[0,2] * 2 / w + theta[0,0] + theta[0,1] - 1
    theta[1,0] = meta[1,0] * w / h
    theta[1,1] = meta[1,1]
    theta[1,2] = meta[1,2] * 2 / h + theta[1,0] + theta[1,1] - 1
    return theta

def calculate_M_from_theta(
        theta: Tensor_grid_M,
        h: int,
        w: int,
        device: torch.device=None,
        inversed: bool=True
    ) -> Tensor_cv2_affine_M:
    """Converts PyTorch theta parameters to OpenCV-style affine matrix.

    Args:
        theta: PyTorch theta parameters from affine_grid.
        h: Height of the image.
        w: Width of the image.
        device: Optional torch device to place the output tensor on.
        inversed: Whether to compute the inverse transformation matrix.

    Returns:
        A tensor representing the OpenCV-style affine transformation matrix.
    """
    if device is None:
        device = theta.device
    
    if not inversed:
        tensor_M = torch.eye(3)
        tensor_M[0:2, :] = theta
        theta = torch.inverse(tensor_M)[0:2, :]

    meta = torch.zeros_like(theta).to(device)
    meta[0, 0] = theta[0, 0]
    meta[0, 1] = theta[0, 1] * (w / h)
    meta[0, 2] = ((theta[0, 2] - theta[0, 0] - theta[0, 1] + 1) * (w / 2))
    
    meta[1, 0] = theta[1, 0] * (h / w)
    meta[1, 1] = theta[1, 1]
    meta[1, 2] = ((theta[1, 2] - theta[1, 0] - theta[1, 1] + 1) * (h / 2))
    
    return meta


def tfrs_inv(
        tensor_tfrs: Tensor_tfrs|Tensor_trs,
        device: torch.device=None
    ) -> Tensor_tfrs:
    """Computes the inverse transformation of a TFRS (Translation, Flip, Rotation, Scale) transform.

    The transformation follows the order: T @ F @ R @ S, where:
    - T: Translation
    - F: Flip
    - R: Rotation
    - S: Scale

    Args:
        tensor_tfrs: A tensor containing transformation parameters [angle, tx, ty, sx, sy] or
            [angle, tx, ty, sx, sy, fx, fy] where:
            - angle: Rotation angle in radians [-π, π]
            - tx, ty: Translation in normalized coordinates [-1, 1]
            - sx, sy: Scale factors in log space
            - fx, fy: Optional flip parameters (0 or 1)
        device: Optional torch device to place the output tensor on.

    Returns:
        A tensor containing the inverse transformation parameters in the same format as input.
    """
    
    theta,tx,ty,sx,sy = tensor_tfrs[0:5]
    fx, fy = torch.ones(1)[0], torch.ones(1)[0]
    if device is None:
        device = tensor_tfrs.device

    if tensor_tfrs.size(0) == 7:
        fx, fy = tensor_tfrs[5:]
        
    fx, fy = fx.to(device), fy.to(device)
    zeros = torch.zeros(1)[0].to(device)
    ones = torch.ones(1)[0].to(device)
    
        
    # Scaling (S^-1)
    sx, sy = 1.0/torch.exp(sx), 1.0/torch.exp(sy)
    S_inv = torch.stack([
        torch.stack([1 / sx, zeros, zeros], dim=-1),
        torch.stack([zeros, 1 / sy, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)

    # Rotation (R^-1)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    R_inv = torch.stack([
        torch.stack([cos_theta, sin_theta, zeros], dim=-1),
        torch.stack([-sin_theta, cos_theta, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)

    # Flip (F^-1)
    F_inv = torch.stack([
        torch.stack([1 / fx if fx != 0 else 0, zeros, zeros], dim=-1),
        torch.stack([zeros, 1 / fy if fy != 0 else 0, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)

    # Translation (T^-1)
    T_inv = torch.stack([
        torch.stack([ones, zeros, -tx], dim=-1),
        torch.stack([zeros, ones, -ty], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)

    # Combine SRFT
    tensor_tfrs_inv = S_inv @ R_inv @ F_inv @ T_inv

    return tensor_tfrs_inv


def _grid_M_to_tfrs_core(grid_M, fx=1, fy=1, device=None):
    # Calculate the rotation angle
    theta = torch.atan2(fy*grid_M[1, 0], fx*grid_M[0, 0])

    # Calculate the scale factors
    sx = torch.sqrt(grid_M[0, 0]**2 + grid_M[1, 0]**2)
    sy = torch.sqrt(grid_M[0, 1]**2 + grid_M[1, 1]**2)
    
    # Calculate the translation components
    tx = grid_M[0, 2]
    ty = grid_M[1, 2]
    if fx < 0 and fy < 0:
        fx, fy = 1, 1
        angle = torch.rad2deg(theta)
        best_angle = (angle + 180) % 360 - 180
        theta = torch.deg2rad(best_angle)
    
    fx, fy = torch.tensor(fx).to(grid_M.device), torch.tensor(fy).to(grid_M.device)
    
    return torch.stack([theta, tx, ty, torch.log(1/sx), torch.log(1/sy), fx, fy], dim=-1).to(device)


def grid_M_to_tfrs(grid_M, fx=1, fy=1, device=None):
    """Decomposes an affine transformation matrix into TFRS parameters.

    Extracts Translation, Flip, Rotation, and Scale parameters from a given affine
    transformation matrix. The transformation is assumed to be in the order T @ F @ R @ S.

    Args:
        grid_M: PyTorch affine transformation matrix of shape [2, 3].
        fx: Initial flip parameter for x-axis (1 or -1).
        fy: Initial flip parameter for y-axis (1 or -1).
        device: Optional torch device to place the output tensor on.

    Returns:
        A tensor containing [angle, tx, ty, sx, sy, fx, fy] parameters.
    """
    # Calculate the rotation angle
    min_diff = float('inf')  # Initialize with a large value
    best_tfrs = None
    for fx in [1, -1]:
        for fy in [1, -1]:
            tfrs_ = _grid_M_to_tfrs_core(grid_M=grid_M, fx=fx, fy=fy, device=device)
            grid_M_ = tfrs_to_grid_M(tfrs_)
            current_diff = torch.sum(torch.abs(grid_M - grid_M_)).item()
            
            if current_diff < min_diff:
                min_diff = current_diff
                best_tfrs = tfrs_
    
    return best_tfrs


def tfrs_to_grid_M(tensor_tfrs: Tensor_tfrs|Tensor_trs, device: torch.device=None) -> Tensor_grid_M:
    """Converts TFRS parameters to an affine transformation matrix.

    Combines Translation, Flip, Rotation, and Scale parameters into a single
    affine transformation matrix in the order T @ F @ R @ S.

    Args:
        tensor_tfrs: A tensor containing either:
            - [angle, tx, ty, sx, sy] for TRS transform
            - [angle, tx, ty, sx, sy, fx, fy] for TFRS transform
        device: Optional torch device to place the output tensor on.

    Returns:
        A tensor of shape [2, 3] representing the affine transformation matrix.
    """
    
    theta,tx,ty,sx,sy = tensor_tfrs[0:5]
    fx, fy = torch.ones(1)[0], torch.ones(1)[0]
    if device is None:
        device = tensor_tfrs.device

    if tensor_tfrs.size(0) == 7:
        fx, fy = tensor_tfrs[5:]
        
    fx, fy = fx.to(device), fy.to(device)
    zeros = torch.zeros(1)[0].to(device)
    ones = torch.ones(1)[0].to(device)
        

    TF = torch.stack([
        torch.stack([fx, zeros, tx], dim=-1),
        torch.stack([zeros, fy, ty], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=0).to(device)

    R = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta), zeros], dim=-1),
        torch.stack([torch.sin(theta), torch.cos(theta), zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)
    
    S = torch.stack([
        torch.stack([1.0/torch.exp(sx), zeros, zeros], dim=-1),
        torch.stack([zeros, 1.0/torch.exp(sy), zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ]).to(device)
    
    grid_M = TF @ R @ S
    return grid_M[0:2, :]
    


def resample_displacement_field_to_size(
        displacement_field: Tensor_disp_2d,
        new_size: torch.Tensor, 
        mode: str='bilinear'
    ) -> Tensor_disp_2d:
    """Resamples a displacement field to a new size.

    Args:
        displacement_field: A tensor of shape (N, H, W, 2) representing the displacement field.
        new_size: Target size for resampling.
        mode: Interpolation mode ('bilinear' by default).

    Returns:
        A tensor of shape (N, new_H, new_W, 2) representing the resampled displacement field.
    """
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), size=new_size, mode=mode, align_corners=False).permute(0, 2, 3, 1)


def get_sampling_grid(tensor_tfrs: Tensor_tfrs, tensor_size: Tuple[int, int], displacement_field: Tensor_disp_2d, device: torch.device) -> Tensor_grid_2d:
    """Generates a sampling grid combining affine transformation and displacement field.

    Args:
        tensor_tfrs: TFRS parameters for affine transformation.
        tensor_size: Target size (height, width) for the sampling grid.
        displacement_field: Additional displacement field to be applied.
        device: Torch device to place the output tensor on.

    Returns:
        A tensor of shape (N, H, W, 2) representing the combined sampling grid.
    """
    # Create affine transformation grid
    grid_M = tfrs_to_grid_M(tensor_tfrs, device=device)
    grid = F.affine_grid(grid_M.unsqueeze(0), [1, 1, tensor_size[0], tensor_size[1]], align_corners=True)

    # Resample displacement field and create sampling grid
    displacement_field = resample_displacement_field_to_size(displacement_field, (grid.size(1), grid.size(2)))
    sampling_grid = grid + displacement_field
    return sampling_grid

def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    """Creates a 2D Gaussian kernel for displacement field smoothing.

    Args:
        kernel_size: Size of the kernel (default: 5).
        sigma: Standard deviation of the Gaussian distribution (default: 1.0).

    Returns:
        A tensor of shape (1, 1, kernel_size, kernel_size) representing the Gaussian kernel.
    """
    x = torch.arange(kernel_size) - kernel_size // 2
    gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
    gaussian_1d /= gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    return gaussian_2d.view(1, 1, kernel_size, kernel_size)


def smooth_displacement_with_kernel(disp, kernel):
    """Smooths a displacement field using a pre-computed kernel.

    Args:
        disp: A tensor of shape (N, H, W, 2) representing the displacement field.
        kernel: A tensor of shape (1, 1, k, k) representing the pre-computed smoothing kernel.

    Returns:
        A tensor of shape (N, H, W, 2) representing the smoothed displacement field.
    """
    disp = disp.permute(0, 3, 1, 2)
    disp_x = F.conv2d(disp[:, 0:1], kernel, padding=kernel.size(-1) // 2)
    disp_y = F.conv2d(disp[:, 1:2], kernel, padding=kernel.size(-1) // 2)
    smoothed_disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)
    return smoothed_disp
