import torch
from abc import ABC, abstractmethod
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy


class Grid2DModelDual(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.tensor_trs = None
        self.displacement_field = None
        self.tensor_size = None
        self.regularization_function = None
        self.dev = None
    
    @abstractmethod
    def forward(self, image: Tensor_image_NCHW, kpts: Tensor_kpts_N_xy, x_ref: Tensor_image_NCHW=None):
        return None
    
    @abstractmethod
    def backforward(self, image: Tensor_image_NCHW, kpts: Tensor_kpts_N_xy, x_ref: Tensor_image_NCHW=None):
        return None
    
    @abstractmethod
    def set_device(self, dev: torch.device):
        """
        """
        return None

    @abstractmethod
    def get_device(self) -> torch.device:
        """
        """
        return None
    
    @abstractmethod
    def regularization():
        return None
    
    @abstractmethod
    def freeze_layer(fwd:bool, inv:bool):
        return None

    @abstractmethod
    def unfreeze_layer(fwd:bool, inv:bool):
        return None

