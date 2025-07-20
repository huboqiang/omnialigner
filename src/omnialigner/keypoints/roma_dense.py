import torch.nn as nn
from typing import Tuple, List, Dict, Any
import dask.array as da
import torch
from torchvision import transforms
import omnialigner as om
from omnialigner.dtypes import KeypointDetectorMeta, Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed, Np_image_HWC, Dask_image_HWC



def get_tuple_transform_ops(resize=(1024, 1024)):
    ops = []
    ops.append(lambda x: torch.from_numpy(x.transpose((2, 0, 1))).float() / 255.0)
    ops.append(transforms.Resize(resize))
    ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)

transform = get_tuple_transform_ops()

class RomaDenseDetector(nn.Module, KeypointDetectorMeta):
    def __init__(self, **kwargs):
        super().__init__()
        from omnialigner.vendor.RoMa.romatch import roma_outdoor
        
        size_coarse = 560
        self.device = kwargs.get("device", torch.device("cpu"))
        upsample_res = kwargs.get("upsample_res", (1024, 1024))
        self.roma = roma_outdoor(device=self.device, coarse_res=size_coarse, upsample_res=upsample_res)
        self.transform = get_tuple_transform_ops(upsample_res)

    def set_upsample_res(self, upsample_res: Tuple[int, int]):
        self.roma.upsample_res = upsample_res
        self.transform = get_tuple_transform_ops(upsample_res)

    def detect(self, image: Tensor_image_NCHW) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed]:
        return None

    def match(self, image_F: Tensor_image_NCHW|Dask_image_HWC, image_M: Tensor_image_NCHW|Dask_image_HWC, method: str="dense") -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw, List[int], Dict[str, Any]]:
        if method == "dense":
            if isinstance(image_F, Tensor_image_NCHW):
                image_F = da.from_array(om.tl.tensor2im(image_F))

            H_A, W_A, _ = image_F.shape
            image_F = self.transform(image_F.compute()).unsqueeze(0)


            if isinstance(image_M, Tensor_image_NCHW):
                image_M = da.from_array(om.tl.tensor2im(image_M))

            H_B, W_B, _ = image_M.shape
            image_M = self.transform(image_M.compute()).unsqueeze(0)


            batch = {"im_A": image_F, "im_B": image_M}
            warp, certainty, corresps = self.roma.match_batched(batch, device=self.device)
            matches, certainty = self.roma.sample(warp, certainty)
            kpts1, kpts2 = self.roma.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    
            return kpts1, kpts2, range(len(kpts1)), corresps
        else:
            raise ValueError(f"method {method} not supported")


    def forward(self, image_F: Tensor_image_NCHW, image_M: Tensor_image_NCHW, method="dense") -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw, List[int], Np_image_HWC]:
        return self.match(image_F, image_M, method)
    