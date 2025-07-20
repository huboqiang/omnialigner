import torch
from omnialigner.utils.point_transform import calculate_rtre

def l1_mean(x: torch.Tensor, p: float=1.0):
    return torch.nn.functional.l1_loss(x**p, torch.zeros_like(x))