from abc import ABC, abstractmethod
from typing import List, Tuple
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy, Np_image_HWC, Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed


class KeypointDetectorMeta(ABC):
    @abstractmethod
    def forward(self, image_F: Tensor_image_NCHW, image_M: Tensor_image_NCHW) -> Tuple[Tensor_kpts_N_xy, Tensor_kpts_N_xy, List[int], Np_image_HWC]:
        pass

    @abstractmethod
    def detect(self, image: Tensor_image_NCHW) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed]:
        """
        Detect keypoints in an image
        Args:
            image (Tensor_image_NCHW): 
        Returns:
            Tensor_kpts_N_xy_raw (Tensor_kpts_N_xy_raw): 
            Tensor_kpts_N_embed (Tensor_kpts_N_embed): 
        """
        pass

    @abstractmethod
    def match(self, image_F: Tensor_image_NCHW, image_M: Tensor_image_NCHW, **kwargs) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw, List[int], Np_image_HWC]:
        """
        Match keypoints between two images
        Args:
            image_F (Tensor_image_NCHW): 
            image_M (Tensor_image_NCHW): 
        Returns:
            tensor_kpts_F (Tensor_kpts_N_xy_raw): 
            tensor_kpts_M (Tensor_kpts_N_xy_raw): 
            l_idxs (List[int]): 
            canvas (np_image_HWC): 
        """
        pass
    
