import math
from typing import Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

from omnialigner.dtypes import Tensor_image_NCHW, Tensor_l_kpt_pair

def cosine_loss(tensor1: Tensor_image_NCHW, tensor2: Tensor_image_NCHW) -> torch.tensor:
    """
    Computes the cosine similarity loss between two NCHW tensors.
    
    Args:
        tensor1 (torch.Tensor): A tensor of shape (N, C, H, W)
        tensor2 (torch.Tensor): A tensor of shape (N, C, H, W)
        
    Returns:
        torch.Tensor: A scalar tensor representing the cosine similarity loss.
    """
    tensor1_flat = tensor1.view(tensor1.size(0), tensor1.size(1), -1)
    tensor2_flat = tensor2.view(tensor2.size(0), tensor2.size(1), -1)
    
    cosine_sim = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=2)
    
    loss = 1 - cosine_sim.mean()
    
    return loss

class GradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(GradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        loss = grad_loss(input, self.penalty)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        dice_loss_ = dice_loss(input, target)
        return dice_loss_
    

# class LossFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lambda_value = 0.2

#     def forward(self, predicted_image: Tensor_image_NCHW, ground_truth_image: Tensor_image_NCHW):
#         """
#         L = (1 − 𝜆)L1 + 𝜆LD-SSIM
#         predicted_image: (B, C, H, W) or (C, H, W)
#         ground_truth_image: (B, C, H, W) or (C, H, W)
#         """
#         if predicted_image is None or ground_truth_image is None:
#             return torch.tensor(0.)

#         if len(predicted_image.shape) == 3:
#             predicted_image = predicted_image.unsqueeze(0)
#         if len(ground_truth_image.shape) == 3:
#             ground_truth_image = ground_truth_image.unsqueeze(0)

#         # func_L1 = DiceLoss()
#         func_L1 = nn.L1Loss()
#         # func_L1 = ncc_local
#         L_DISS = func_L1(predicted_image, ground_truth_image)
#         LD_SSIM = 1 - ssim(predicted_image, ground_truth_image,
#                            data_range=1, size_average=True)
        
#         L = (1 - self.lambda_value) * L_DISS + self.lambda_value * LD_SSIM
#         return L

class LossFunction2D(nn.Module):
    def __init__(self, func_dict=None):
        super().__init__()
        if func_dict is None:
            func_dict = {
                nn.functional.l1_loss: {"weight": 0.8, "params" : {}},
                ssim_loss:  {"weight": 0.2, "params" : {}},
            }
        self.dict_loss_funcs = func_dict

    def forward(self, predicted_image: Tensor_image_NCHW, ground_truth_image: Tensor_image_NCHW):
        """
        Calculate 2D loss for a pair of images. Image pair could be generated via utils.image_pad.pad_tensors from two raw images
        
        Args:
            tensor1 (torch.Tensor): A tensor of shape (N, C, H, W)
            tensor2 (torch.Tensor): A tensor of shape (N, C, H, W)
            
        Returns:
            torch.Tensor: A scalar tensor representing the cosine similarity loss.
        """
        total_loss = 0.0
        if predicted_image is None or ground_truth_image is None:
            return total_loss

        if len(predicted_image.shape) == 3:
            predicted_image = predicted_image.unsqueeze(0)
        if len(ground_truth_image.shape) == 3:
            ground_truth_image = ground_truth_image.unsqueeze(0)

        for func, element in self.dict_loss_funcs.items():
            if type(func) is str:
                func = globals()[func]
            
            total_loss += element["weight"] * func(predicted_image, ground_truth_image, **element["params"])

        return total_loss


class LossFunc3D(nn.Module):
    def __init__(self, func_dict=None, max_k=9, batch_size=32):
        """
        3D loss means given Z stacked images [Z, C, H, W], calculated avg loss for ([i, i+1], [i, i+2], ... [i, i+k]) pairs in Z-axis

        Args:
            max_k (int, optional): maxium distance in Z. Defaults to 9.
            batch_size (int, optional): batch size could avoiding OOM . Defaults to 32.
        """
        super().__init__()
        self.max_k = max_k
        self.batch_size = batch_size
        self.loss_func = LossFunction2D(func_dict=func_dict)

    def forward(self, image_3d_tensor: Tensor_image_NCHW) -> torch.tensor:
        """
        Calculate 3D loss for a pair of images. A list of images could be generated via utils.image_pad.pad_tensors from two raw images
        
        Args:
            3D tensor (torch.Tensor): A tensor of shape (Z, C, H, W)
            
        Returns:
            torch.Tensor: A scalar tensor of 3D loss
        """
        N = image_3d_tensor.size(0)
        total_loss = 0.0

        # Process each distance k
        for k in range(1, self.max_k + 1):
            start_idx = 0
            while start_idx + k < N:
                end_idx = min(start_idx + self.batch_size, N - k)
                predicted_images = image_3d_tensor[start_idx:end_idx]
                ground_truth_images = image_3d_tensor[start_idx + k:end_idx + k]
                loss = self.loss_func(predicted_images, ground_truth_images)
                total_loss += loss.sum() * (end_idx-start_idx)
                start_idx += self.batch_size

        return total_loss / N


# class NCCLocal(nn.Module):
#     def __init__(self):
#         super(NCCLocal, self).__init__()
    
#     def forward(self, tensor_F, tensor_M, **cost_function_params):
#         if tensor_F is None or tensor_M is None:
#             return torch.tensor(0.)
        
#         return ncc_local(tensor_F, tensor_M, **cost_function_params)

def ssim_loss(predicted_image: Tensor_image_NCHW, ground_truth_image: Tensor_image_NCHW):
    LD_SSIM = 1 - ssim(predicted_image, ground_truth_image,
                           data_range=1, size_average=True)
    return LD_SSIM
        

def abs_loss(grid, penalty="l1"):
    """
    Computes the gradient loss of pytorch grid (see F.grid_sample)
    
    Args:
        grid (torch.Tensor): 2D grid-tensor of shape (N, 2, H, W)
        
    Returns:
        torch.Tensor: A scalar tensor representing the gradient loss.
    """    
    if penalty == "l1":
        return torch.nn.functional.l1_loss(grid, torch.zeros_like(grid))
    elif penalty == "l2":
        return torch.nn.functional.mse_loss(grid, torch.zeros_like(grid))
    
    raise ValueError(f"Unknown penalty type: {penalty}. Supported types are 'l1' and 'l2'.")

def grad_loss(grid, penalty="l1"):
    """
    Computes the gradient loss of pytorch grid (see F.grid_sample)
    
    Args:
        grid (torch.Tensor): 2D grid-tensor of shape (N, 2, H, W)
        
    Returns:
        torch.Tensor: A scalar tensor representing the gradient loss.
    """    
    dH = (grid[:, :, 1:, :] - grid[:, :, :-1, :])
    dW = (grid[:, :, :, 1:] - grid[:, :, :, :-1])
    if penalty == "l2":
        dH = dH * dH
        dW = dW * dW
        
    std_ = torch.std(dH, dim=(2,3)) + torch.std(dW, dim=(2,3))
    loss = torch.mean(std_)
    return loss


def dice_loss(input, target):
    """
    Calculate Dice Loss

    Args:
        input (torch.Tensor): A binary tensor of shape (H, W)
        target (torch.Tensor): A binary tensor of shape (H, W)

    Return:
        torch.Tensor: A scalar tensor representing the Dice loss.
    """
    assert input.size() == target.size(), "Input and target masks should have the same shape"

    input_flat = input.view(-1)
    target_flat = target.view(-1)

    intersection = (input_flat * target_flat).sum()
    dice_score = (2. * intersection + 1.) / (input_flat.sum() + target_flat.sum() + 1.)
    dice_loss_ = 1 - dice_score

    return dice_loss_

def loss_with_attr(x, y, attr=torch.tensor(1.), scale=False, H=1600, W=1600, lamb_attr=0.1):
    """
    
    """
    if scale:
        H_center, W_center = H // 2, W // 2
        x = (x-W_center) / W_center
        y = (y-H_center) / H_center
    
    # dist = torch.mean( attr*(x - y) ** 2)
    dist = torch.mean(attr*torch.abs(x - y))
    reg = 1 - torch.mean(attr)
    loss = dist + lamb_attr*reg
    return loss

def loss_kpts_attr_multi_layers_bak(l_kpts_pair):
    """
    Calculate keypoint different loss for list of key-point pairs.

    Args:
        l_kpts_pair:    [ kpt_pair1, kpt_pair2, ... ]
            kpt_pair (scaled_F, scaled_M):  
                scaled_F    [x/W, y/H] torch coords
                scaled_M    [x/W, y/H] torch coords                
        
        target (torch.Tensor): A binary tensor of shape (H, W)

    Return:
        torch.Tensor: A scalar tensor representing the Dice loss.
    """
    loss = []
    for idx,pair in enumerate(l_kpts_pair):
        if len(pair) < 2 or pair[0] is None or pair[1] is None:
            continue
        
        loss_ = F.l1_loss(pair[0].float(), pair[1].float())
        loss.append(loss_)

    if len(loss) == 0:
        return torch.tensor(0.)

    losses = torch.stack(loss)
    total_loss = torch.mean(losses)
    return total_loss


def loss_kpts_attr_multi_layers_tre(
        l_kpts_pair: Tensor_l_kpt_pair, 
        **kwargs
    ) -> torch.Tensor:
    """
    Calculate keypoint different loss for list of key-point pairs.

    Args:
        l_kpts_pair:    N layers of layer_kpt_pair
            layer_kpt_pair:
                lists of kpt_pair for [layer_i,layer_i+1], [layer_i, layer_i+2]
                kpt_pair (scaled_F, scaled_M):  
                    scaled_F    [x/W, y/H] torch coords
                    scaled_M    [x/W, y/H] torch coords
        
        target (torch.Tensor): A binary tensor of shape (H, W)

    Return:
        torch.Tensor: A scalar tensor representing the kpt loss
    """
    loss = []
    for idx, layer_kpt_pair in enumerate(l_kpts_pair):
        for idx_pair, pair in enumerate(layer_kpt_pair):
            if len(pair) < 2 or pair[0] is None or pair[1] is None:
                continue

            if pair[0].shape[0] < 1 or pair[1].shape[0] < 1 or pair[0].shape[1] < 1 or pair[1].shape[1] < 1:
                continue
            

            loss_ = nn.functional.l1_loss(pair[0].float(), pair[1].float())
            loss.append(loss_)

    if len(loss) == 0:
        return torch.tensor(0.)

    losses = torch.stack(loss)
    total_loss = torch.mean(losses)
    return total_loss


def loss_kpts_attr_multi_layers(
        l_kpts_pair: Tensor_l_kpt_pair,
        alpha: float = 0.1
    ) -> torch.Tensor:
    """
    Calculate keypoint different loss for list of key-point pairs.

    Args:
        l_kpts_pair:    N layers of layer_kpt_pair
            layer_kpt_pair:
                lists of kpt_pair for [layer_i,layer_i+1], [layer_i, layer_i+2]
                kpt_pair (scaled_F, scaled_M):  
                    scaled_F    [x/W, y/H] torch coords
                    scaled_M    [x/W, y/H] torch coords
        
        alpha:        aTRE weight

    Return:
        torch.Tensor: A scalar tensor representing the kpt loss
    """
    loss_list = []
    TRE_list = []
    device = l_kpts_pair[0][0][0].device if len(l_kpts_pair) > 0 and len(l_kpts_pair[0]) > 0 else torch.device('cpu')
    cumerror = torch.zeros(2, device=device)

    for layer_kpt_pair in l_kpts_pair:
        Y_errors = []
        X_errors = []
        
        for pair in layer_kpt_pair:
            if len(pair) < 2 or pair[0] is None or pair[1] is None:
                continue

            if pair[0].shape[0] < 1 or pair[1].shape[0] < 1 or pair[0].shape[1] < 2 or pair[1].shape[1] < 2:
                continue

            loss_ = nn.functional.l1_loss(pair[0].float(), pair[1].float())
            loss_list.append(loss_)
            
            Y_error = torch.mean(pair[1][:, 1] - pair[0][:, 1])
            X_error = torch.mean(pair[1][:, 0] - pair[0][:, 0])
            Y_errors.append(Y_error)
            X_errors.append(X_error)
        
        if Y_errors and X_errors:
            mean_Y_error = torch.nanmean(torch.stack(Y_errors))
            mean_X_error = torch.nanmean(torch.stack(X_errors))
            tre = torch.sqrt(mean_Y_error**2 + mean_X_error**2)
            TRE_list.append(tre)

    mATRE = torch.nanmean(torch.stack(TRE_list)) if TRE_list else torch.tensor(0., device=device)
    total_L1_loss = torch.nanmean(torch.stack(loss_list)) if loss_list else torch.tensor(0., device=device)    
    total_loss = total_L1_loss + alpha * mATRE
    return total_loss

def ncc_local(
    sources: torch.Tensor,
    targets: torch.Tensor,
    device: Union[str, torch.device, None]=None,
    **params : dict) -> torch.Tensor:
    """
    Local normalized cross-correlation (as cost function) using PyTorch tensors.

    Implementation inspired by VoxelMorph (with some modifications).

    Parameters
    ----------
    sources : torch.Tensor(Bx1xMxN)
        The source tensor
    targest : torch.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    
    """
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size = params['win_size']
    except:
        win_size = 3
    try:
        mask = params['mask']
    except:
        mask = None
    window = (win_size, ) * ndim
    
    n_channels = sources.shape[1]
    if device is None:
        sum_filt = torch.ones([1, n_channels, *window]).type_as(sources) / n_channels
    else:
        sum_filt = torch.ones([1, n_channels, *window], device=device) / n_channels

    if mask is not None:
        targets = targets * mask

    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -torch.mean(ncc)


def ncc_global(
    sources: torch.Tensor,
    targets: torch.Tensor,
    device: Union[str, torch.device, None]="cpu",
    **params : dict) -> torch.Tensor:
    """
    Global normalized cross-correlation (as cost function) using PyTorch tensors.

    Parameters
    ----------
    sources : torch.Tensor(Bx1xMxN)
        The source tensor
    targest : torch.Tensor (Bx1xMxN)
        The target target
    device : str
        The device where source/target are placed
    params : dict
        Additional cost function parameters

    Returns
    ----------
    ncc : float
        The negative of normalized cross-correlation (average across batches)
    """
    sources = (sources - torch.min(sources)) / (torch.max(sources) - torch.min(sources))
    targets = (targets - torch.min(targets)) / (torch.max(targets) - torch.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    size = sources.size()
    prod_size = torch.prod(torch.Tensor(list(size[1:])))
    sources_mean = torch.mean(sources, dim=list(range(1, len(size)))).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_mean = torch.mean(targets, dim=list(range(1, len(size)))).view((targets.size(0),) + (len(size)-1)*(1,))
    sources_std = torch.std(sources, dim=list(range(1, len(size))), unbiased=False).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_std = torch.std(targets, dim=list(range(1, len(size))), unbiased=False).view((targets.size(0),) + (len(size)-1)*(1,))
    ncc = (1 / prod_size) * torch.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std), dim=list(range(1, len(size))))
    ncc = torch.mean(ncc)
    if ncc != ncc:
        ncc = torch.autograd.Variable(torch.Tensor([-1]), requires_grad=True).to(device)
    return -ncc

def diffusion_relative(
    displacement_field: torch.Tensor,
    **params : dict) -> torch.Tensor:
    """
    Relative diffusion regularization (with respect to the input size) (PyTorch).
    Parameters
    ----------
    displacement_field : torch.Tensor
        The input displacment field (2-D or 3-D) (B x size x ndim)
    params : dict
        Additional parameters
    Returns
    ----------
    diffusion_reg : float
        The value denoting the decrease of displacement field smoothness
    """
    ndim = len(displacement_field.size()) - 2
    if ndim == 2:
        dx = ((displacement_field[:, 1:, :, :] - displacement_field[:, :-1, :, :])*displacement_field.shape[1])**2
        dy = ((displacement_field[:, :, 1:, :] - displacement_field[:, :, :-1, :])*displacement_field.shape[2])**2
        diffusion_reg = (torch.mean(dx) + torch.mean(dy)) / 2
    elif ndim == 3:
        dx = ((displacement_field[:, 1:, :, :, :] - displacement_field[:, :-1, :, :, :])*displacement_field.shape[1])**2
        dy = ((displacement_field[:, :, 1:, :, :] - displacement_field[:, :, :-1, :, :])*displacement_field.shape[2])**2
        dz = ((displacement_field[:, :, :, 1:, :] - displacement_field[:, :, :, :-1, :])*displacement_field.shape[3])**2
        diffusion_reg = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3
    else:
        raise ValueError("Unsupported number of dimensions.")
    return diffusion_reg

