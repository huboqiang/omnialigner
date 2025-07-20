from abc import ABC, abstractmethod
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_disp_2d

class Disp2DModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.displacement_field = None
        self.tensor_size = None
        self.flow_model = None
        self.dev = None
    
    @abstractmethod
    def forward(self, displacement_field: Tensor_disp_2d, image_current: Tensor_image_NCHW, image_next: Tensor_image_NCHW) -> Tensor_disp_2d:
        pass
