# import pdb
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import OrderedDict

from tqdm import tqdm
import yaml
import dask.array as da
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import omnialigner as om
from omnialigner.omni_3D import Omni3D
from omnialigner.align.models.loss import LossFunc3D, loss_kpts_attr_multi_layers
from omnialigner.dtypes import Grid2DModelDual
from omnialigner.logging import logger as logging
from omnialigner.plotting.keypoint_viz import plot_align_batch, plot_ovlp_kpts
from omnialigner.datasets import OverlappedImageLayerDataset
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_trs, Tensor_kpts_N_xy, Tensor_kpt_pair, Tensor_l_kpt_pair
from omnialigner.utils.config import load_from_string
from omnialigner.utils.image_transform import create_pyramid
from omnialigner.metrics.regbenchmark_py import benchmark_kpts


def select_keypoints(
        kpt_pairs: Tensor_l_kpt_pair,
        crop_region: Tuple[float, float, float, float]=(0., 1., 0., 1.),
        scale: Tuple[float, float]=(1., 1.)
    ) -> Tensor_l_kpt_pair:
    """
    Crop the keypoints in given region and scale them to the new size.
    Args:
        kpt_pairs: Tensor_l_kpt_pair: List of keypoint pairs.
        crop_region: Tuple[float, float, float, float]: Crop region. [i_beg, i_end, j_beg, j_end] in [0, 1]
        scale: Tuple[float, float]: Scale of the keypoints. [h_raw / h_crop, w_raw / w_crop]
    """
    l_kpt_pairs = []
    for layer_kpt_pair in kpt_pairs:
        layer_kpt_pair_cropped = []
        for kpt_pair in layer_kpt_pair:
            idx_F = (kpt_pair[0][:, 1] > crop_region[0]) & (kpt_pair[0][:, 1] < crop_region[1]) & (kpt_pair[0][:, 0] > crop_region[2]) & (kpt_pair[0][:, 0] < crop_region[3])
            idx_M = (kpt_pair[1][:, 1] > crop_region[0]) & (kpt_pair[1][:, 1] < crop_region[1]) & (kpt_pair[1][:, 0] > crop_region[2]) & (kpt_pair[1][:, 0] < crop_region[3])

            idx = idx_F & idx_M
            kpt_pair[0] = (kpt_pair[0][idx].detach().cpu() - torch.tensor([crop_region[2], crop_region[0]])) * scale[0]
            kpt_pair[1] = (kpt_pair[1][idx].detach().cpu() - torch.tensor([crop_region[2], crop_region[0]])) * scale[1]
            layer_kpt_pair_cropped.append(kpt_pair)
        l_kpt_pairs.append(layer_kpt_pair_cropped)
    return l_kpt_pairs



def apply_disp_to_HD(om_data: Omni3D, l_layers: List[int], crop_region: Tuple[float, float, float, float], dict_model: Dict[str, Any]):
    om_data.set_tag("gray")
    om_data.set_zoom_level(0)
    da_arr_nchw = om_data.load_3d_NCHW(aligned_tag="AFFINE_HD", l_layers=l_layers)
    l_imgs = []
    for idx, i_layer in tqdm(enumerate(l_layers)):
        da_arr = da.moveaxis(da_arr_nchw[idx, :, :, :], 0, -1)
        img = _apply_disp_to_HD(da_arr, i_layer, crop_region, dict_model)
        l_imgs.append(img)
    return l_imgs

def _apply_disp_to_HD(da_arr, i_layer, crop_region, dict_model):
    H, W = da_arr.shape[0], da_arr.shape[1]
    i_beg_raw = int(crop_region[0] * H)
    i_end_raw = int(crop_region[1] * H)
    j_beg_raw = int(crop_region[2] * W)
    j_end_raw = int(crop_region[3] * W)

    disp_field = dict_model[f"{i_layer}.displacement_field"]
    disp_field_large = om.tl.resize_grid(disp_field.detach().cpu(), (H, W))
    disp_field_large_crop = disp_field_large[:, i_beg_raw:i_end_raw, j_beg_raw:j_end_raw, :]

    image_crop = da_arr[i_beg_raw:i_end_raw, j_beg_raw:j_end_raw, :]

    zoom_H = H / image_crop.shape[0]
    zoom_W = W / image_crop.shape[1]
    disp_field_large_crop[:, :, :, 0] = disp_field_large_crop[:, :, :, 0] * zoom_H
    disp_field_large_crop[:, :, :, 1] = disp_field_large_crop[:, :, :, 1] * zoom_W
    grid_crop = om.tl.disp2grid(disp_field_large_crop)
    tensor_image_crop = om.tl.im2tensor(image_crop)
    tensor_transformed = F.grid_sample(tensor_image_crop, grid_crop, mode='bilinear', align_corners=True)

    return tensor_transformed


class OmniAligner(nn.Module):
    def __init__(self,
            image_3d_tensor: Tensor_image_NCHW,
            l_kpt_pairs: Tensor_l_kpt_pair= None,
            l_init_trs: List[Tensor_trs]=None,
            dict_config: Dict[str, Any]=None,
            model_type: str="affine",
            save_prefix: str="",
            log_prefix=None,
            **kwargs
        ):
        """
        Omni3DAligner class.

        Args:
            image_3d_tensor: Tensor_image_NCHW
                            The input 3D image tensor.
            l_kpt_pairs: Tensor_l_kpt_pair
                            N layers of layer_kpt_pair
                                layer_kpt_pair:
                                    lists of kpt_pair for [layer_i,layer_i+1], [layer_i, layer_i+2]
                                    kpt_pair (scaled_F, scaled_M):  
                                        scaled_F    [x/W, y/H] torch coords
                                        scaled_M    [x/W, y/H] torch coords

            l_init_trs: List[Tensor_trs]: 
                            List of initial transformations.
            dict_config: Dict[str, Any]: 
                            Dictionary of configuration parameters.
            model_type: str: 
                            Type of the model.
            save_prefix: str: 
                            Prefix for saving the model.
            log_prefix: str: 
                            Prefix for logging.
        """
        super().__init__()
        self.l_kpt_pairs = l_kpt_pairs
        self.N = image_3d_tensor.shape[0]
        self.image_3d_tensor = image_3d_tensor
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = dict_config
        for key, value in self.config.items():
            logging.info(f"{key}: {value}")
        self.config_model = dict_config[model_type]
        self.use_pyramid = False
        if "used_levels" in self.config_model:
            self.use_pyramid = True


        
        self.model_type = model_type
        if model_type == "nonrigid_tiles":
            self.model_type = "nonrigid"
        
        model_name = self.config_model["model"]
        self.grid_model: Grid2DModelDual = load_from_string(model_name)

        self.weight_image = self.config_model["weight_image"]
        self.weight_kpts = self.config_model["weight_kpts"]
        self.weight_reg = self.config_model["weight_reg"]
        if "freezed_layers" not in self.config_model:
            self.config_model["freezed_layers"] = []

        self.k = self.config["dataloader"]["k"]
        self.batch_size = self.config["dataloader"]["batch_size"]
        self.overlap = self.config["dataloader"]["overlap"]

        self.save_prefix = save_prefix
        self.log_prefix = self.config["trainer"]["log_prefix"]
        if log_prefix is not None:
            self.log_prefix = log_prefix

        self.current_epoch = 0
        self.i_iter_global = [ 0 for _ in range(self.N) ]
        self.iterations = self.config["trainer"]["iterations"]
        self.n_iters_show = self.config["trainer"]["n_iters_show"]
        self.n_cols = self.config["trainer"]["n_cols"]
        self.figsize = self.config["trainer"]["figsize"]
        self.show_ovlp = self.config_model["show_ovlp"] if "show_ovlp" in self.config_model else True
        self.loss_func_kpt_pairs = loss_kpts_attr_multi_layers
        self.full_figsize = kwargs.get("full_figsize", (20, 20))
        self.full_n_cols = kwargs.get("full_n_cols", 8)
        os.makedirs(self.log_prefix, exist_ok=True)
        self.n_keypoints = self.config_model.get("n_keypoints", -1)
        self.grid2d_modules: List[Grid2DModelDual] = None
        if self.use_pyramid:
            n_epoches = self.config_model["num_levels"]
            n_epoches_used = self.config_model["used_levels"]
            self.target_pyramid = create_pyramid(image_3d_tensor, num_levels=n_epoches)[:n_epoches_used]
            ### INIT grid2d_modules in small sizes
            h, w = self.target_pyramid[0].shape[2], self.target_pyramid[0].shape[3]
            self.tensor_size = [h, w] #self.config_model.get("tensor_size", [h, w])

        self.update_tensor_size(image_3d_tensor=image_3d_tensor)
        if l_init_trs is None:
            l_init_trs = [ torch.FloatTensor([0, 0, 0, 0, 0]) for i in range(self.N)]

        self.grid2d_modules = [
            self.grid_model(
                init_trs=l_init_trs[i],
                current_epoch=0,
                tensor_size=self.tensor_size,
                **self.config_model
            ) for i in range(self.N)
        ]
        _, C, _, _ = image_3d_tensor.shape
        cost_func = None
        if "cost_function" in self.config_model:
            cost_func = self.config_model["cost_function"]
        
        self.loss_func_image3d = LossFunc3D(func_dict=cost_func, max_k=self.k, batch_size=self.batch_size)
        self.writer = SummaryWriter(f'{self.log_prefix}/{model_type}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    def forward(self,
            batch_images: Tensor_image_NCHW,
            indices: List[int],
            use_input_size: bool=False
        ) -> Tensor_image_NCHW:
        """
        Args:
            batch_images (Tensor_image_NCHW): the input images, already sliced by indices [N_beg:N_end, C, H, W]
            indices (List[int]): the indices of the images to be transformed:             range(N_beg, N_end)
            use_input_size (bool): whether to use the input size

        Returns:
            Tensor_image_NCHW: the transformed images
        """
        l_images = []
        if use_input_size:
            for i, index in enumerate(indices):
                out_image = self.grid2d_modules[index].forward_raw_size(batch_images[i:i+1])
                l_images.append(out_image)
        else:
            for i, index in enumerate(indices):
                out_image, _ = self.grid2d_modules[index].forward(batch_images[i:i+1], None)
                l_images.append(out_image)

        # pdb.set_trace()
        transformed_images = torch.stack(l_images).squeeze(1)
        return transformed_images

    def forward_kpts(self,
            l_kpt_pairs: Tensor_l_kpt_pair,
            indices: List[int]
        ) -> Tensor_l_kpt_pair:
        l_kpt_pairs_moved = []
        for i, index in enumerate(indices[:-1]):
            if len(l_kpt_pairs[i]) < 1:
                l_kpt_pairs_moved.append([])
                continue

            layer_kpt_pair_moved = []
            layer_kpt_pair = l_kpt_pairs[i]
            if len(layer_kpt_pair) < 1:
                null_layer = [[torch.tensor([]), torch.tensor([])]]
                l_kpt_pairs_moved.append(null_layer)
                continue

            for i_zdist, kpt_pair in enumerate(layer_kpt_pair):
                if index+i_zdist+1 >= len(self.grid2d_modules):
                    continue

                if len(kpt_pair) < 2:
                    null_kpt_pair = [torch.tensor([]), torch.tensor([])]
                    layer_kpt_pair_moved.append(null_kpt_pair)
                    continue

                kpt_F = kpt_pair[0]
                kpt_M = kpt_pair[1]
                if len(kpt_F.shape) < 2 or kpt_F.shape[1] < 1 or len(kpt_M.shape) < 2 or kpt_M.shape[1] < 1:
                    null_kpt_pair = [torch.tensor([]), torch.tensor([])]
                    layer_kpt_pair_moved.append(null_kpt_pair)
                    continue

                if self.n_keypoints > 0 and kpt_F.shape[0] > self.n_keypoints:
                    indices = torch.randperm(kpt_F.shape[0])[:self.n_keypoints]
                    kpt_F = kpt_F[indices]
                    kpt_M = kpt_M[indices]

                _, kpt_F_moved = self.grid2d_modules[index].forward(None, kpt_F)
                _, kpt_M_moved = self.grid2d_modules[index+i_zdist+1].forward(None, kpt_M)
                layer_kpt_pair_moved.append([kpt_F_moved, kpt_M_moved])
            l_kpt_pairs_moved.append(layer_kpt_pair_moved)

        return l_kpt_pairs_moved


    def training_step(self,
            batch: Tuple[Tensor_image_NCHW, List[List[Tuple[Tensor_kpts_N_xy, Tensor_kpts_N_xy]]], torch.Tensor]
        ) -> torch.Tensor:
        batch_images, l_kpt_pairs, batch_indices = batch
        batch_images = batch_images.squeeze(0)
        batch_indices = batch_indices.squeeze(0)

        self._prepare_grid_module(batch_images, batch_indices)

        iterations = self.iterations
        if "iterations" in self.config_model:
            iterations = self.config_model["iterations"][self.current_epoch]

        total_loss = float('inf')
        batch_images, l_kpt_pairs = self._prepare_device(batch_images=batch_images, l_kpt_pairs=l_kpt_pairs, batch_indices=batch_indices, use_pyramid=self.use_pyramid)
        for mod in self.grid2d_modules:
            mod.set_device(self.dev)

        opt, sched = self._configure_optimizers()
        self.grid2d_modules[batch_indices[-1]].freeze_layer(fwd=False, inv=(self.weight_image == 0))
        if batch_indices[0] > 0:
            l_ovlp_freezed = [batch_indices[idx] for idx in range(min(self.overlap, len(batch_indices)))]
            for index in l_ovlp_freezed:
                grid2d_module = self.grid2d_modules[index]
                grid2d_module.freeze_layer()

        for index in self.config_model["freezed_layers"]:
            # if hasattr(self.grid2d_modules[index], "displacement_field"):
            #     self.grid2d_modules[index].displacement_field = nn.Parameter(torch.zeros_like(self.grid2d_modules[index].displacement_field))

            self.grid2d_modules[index].freeze_layer()

        beg, end = batch_indices[0], batch_indices[-1]        
        for iteration in tqdm(range(iterations), desc=f"epoch {self.current_epoch} iter"):
            opt.zero_grad()
            total_loss, _, _, dict_loss = self.loss3d_per_iterations(batch_images, l_kpt_pairs, batch_indices)

            if total_loss.requires_grad:
                total_loss.backward()

            for loss_name, loss_value in dict_loss.items():
                self.writer.add_scalar(f'train/{loss_name}/{beg}_{end}', loss_value, self.i_iter_global[beg])


            if "trs_save_prefix" in self.config_model:
                l_trs_save_prefix = self.config_model["trs_save_prefix"]
                for index in batch_indices:
                    grid2d_module = self.grid2d_modules[index]
                    for i_rtk, prefix in enumerate(l_trs_save_prefix):
                        self.writer.add_scalar(f'train/{prefix}/{index}/{beg}_{end}', grid2d_module.tensor_trs[i_rtk], self.i_iter_global[beg])

            if self.n_iters_show > 0 and iteration % self.n_iters_show == 0 or iteration == iterations - 1:
                self.viz_images(batch_indices)

            self.i_iter_global[beg] += 1
            opt.step()
            if sched is not None:
                sched.step()

        ## refreezed
        for index in batch_indices:
            grid2d_module = self.grid2d_modules[index]
            grid2d_module.freeze_layer()

        self.post_one_step(batch_indices)
        return total_loss

    def train_epoch(self,
            batch: torch.FloatTensor,
        ) -> float:
        loss = self.training_step(batch)
        return loss.item()

    def loss3d_per_iterations(self,
            batch_images: Tensor_image_NCHW,
            l_kpt_pairs: Tensor_l_kpt_pair,
            batch_indices: List[int]
        ) -> Tuple[torch.Tensor, Tensor_image_NCHW, Tensor_l_kpt_pair, Dict[str, torch.Tensor]]:
        loss_image = torch.tensor(0.)
        loss_kpts = torch.tensor(0.)
        transformed_images = self.forward(batch_images, batch_indices)
        transformed_kpt_pairs = self.forward_kpts(l_kpt_pairs, batch_indices)
        if self.weight_image > 0:
            loss_image = self.loss_func_image3d(transformed_images)

        l_regs = [self.grid2d_modules[index].regularization() for index in batch_indices]
        l1_local_regularization = sum([reg_[0] for reg_ in l_regs]) / len(batch_indices)

        if self.weight_kpts != 0:
            loss_kpts = self.loss_func_kpt_pairs(transformed_kpt_pairs)

        total_loss = self.weight_image * loss_image + \
                     self.weight_kpts * loss_kpts + \
                     self.weight_reg * l1_local_regularization

        # if torch.isnan(total_loss):
        #     pdb.set_trace()

        dict_loss = {
            "loss_3d": total_loss,
            "loss_image": self.weight_image * loss_image,
            "loss_kpts": self.weight_kpts * loss_kpts,
            "loss_reg": self.weight_reg * l1_local_regularization
        }
        return total_loss, transformed_images, transformed_kpt_pairs, dict_loss

    def viz_images(self,
            indices: List[int],
        ) -> Tuple[np.ndarray, Tensor_l_kpt_pair]:
        batch_images = self.image_3d_tensor[indices]
        batch_kpts = self.l_kpt_pairs[indices[0]:indices[-1]]
        transformed_images, l_kpts_moved = self.viz(batch_images, l_kpt_pairs=batch_kpts, start_idx=indices[0], end_idx=indices[-1]+1)
        fig = plot_align_batch(
            transformed_images,
            l_kpts_moved,
            batch_indices=indices,
            n_cols=self.n_cols,
            figsize=self.figsize,
        )
        self.writer.add_figure(f"train/images/{indices[0]}_{indices[-1]}", fig, self.i_iter_global[indices[0]])
        if self.show_ovlp:
            fig.savefig(f"{self.log_prefix}/epoch{self.current_epoch}-{indices[0]}-{indices[-1]}.png")

        return transformed_images, l_kpts_moved

    def post_one_step(self,
            indices: List[int],
        ):
        model_pt_ = f"{self.log_prefix}/epoch{self.current_epoch}-{indices[0]}-{indices[-1]}.ckpt"
        # dict_model = nn.ModuleList(self.grid2d_modules).state_dict()
        dict_model = OrderedDict()
        if self.model_type == "affine":
            for idx in indices:
                dict_model[f"{idx}.tensor_trs"] = self.grid2d_modules[idx].tensor_trs
        elif self.model_type == "nonrigid":
            for idx in indices:
                dict_model[f"{idx}.displacement_field"] = self.grid2d_modules[idx].disp
        else:
            raise ValueError(f"model_type {self.model_type} not supported")
        torch.save(dict_model, model_pt_, _use_new_zipfile_serialization=False)

    def viz(self,
            image_3d_tensor: Tensor_image_NCHW,
            l_kpt_pairs: Tensor_l_kpt_pair=None,
            start_idx: int=0,
            end_idx: int=-1
        ) -> Tuple[Tensor_image_NCHW, Tensor_l_kpt_pair]:
        if end_idx < 0:
            end_idx = self.N

        self.n_keypoints = -1
        with torch.no_grad():
            dict_raw_size = {}
            self.tensor_size = [self.image_3d_tensor.shape[2], self.image_3d_tensor.shape[3]]
            for idx_ in range(start_idx, end_idx):
                raw_size = self.grid2d_modules[idx_].tensor_size
                dict_raw_size[idx_] = raw_size
                self.grid2d_modules[idx_].tensor_size = self.tensor_size

            idx_range = torch.arange(start_idx, end_idx)
            img, l_kpt_pairs = self._prepare_device(
                    batch_images=image_3d_tensor,
                    l_kpt_pairs=l_kpt_pairs,
                    batch_indices=idx_range,
                    use_pyramid=False
            )
            transformed_images = self.forward(img, idx_range)
            l_kpts_moved = None
            if l_kpt_pairs is not None:
                l_kpts_moved = []
                l_kpts_moved_ = self.forward_kpts(l_kpt_pairs, idx_range)
                for layer_kpt_pair in l_kpts_moved_:
                    layer_kpt_pair_moved = []
                    for kpt_pair_ in layer_kpt_pair:
                        kpt_pair = [kpt_pair_[0].detach().cpu(), kpt_pair_[1].detach().cpu()]
                        layer_kpt_pair_moved.append(kpt_pair)
                    l_kpts_moved.append(layer_kpt_pair_moved)

            for idx_ in idx_range:
                self.grid2d_modules[int(idx_)].tensor_size = dict_raw_size[int(idx_)]

        self.n_keypoints = self.config_model.get("n_keypoints", -1)
        return transformed_images.detach().cpu(), l_kpts_moved


    def process_aligned_kpts(self, l_kpt_pairs: List[Tensor_kpts_N_xy|Tensor_kpt_pair], scale_level=1., show_img=False):
        """
            validate the model on the given kpt pairs
            ---
            Args:
                l_kpt_pairs List[Tensor_kpts_N_xy|Tensor_kpt_pair]: 
                            List of Affine transformed
                                kpt (regbenchmark_AccumulatedTRE_linearfit) or 
                                kpt-pairs (regbenchmark_AccumulatedTRE_cumulative)
                scale_level (float): 
                            the scale level to nonrigid kpt. 
                            default: 1.
                show_img (bool): 
                            whether to show the images after transformation
                            default: False

            Returns:
                l_kpt_pairs_scaled (List[Tensor_kpt_pair]): 
                            List of raw size nonrigid kpt-pairs
                l_imgs (List[List[Tensor_image_NCHW]]): 
                            List of images after transformation if show_img is True
        """
        l_imgs = None
        if show_img:
            l_imgs = []
        l_kpt_pairs_scaled = []
        self._prepare_device(batch_images=None, l_kpt_pairs=None, use_pyramid=False)
        for mod in self.grid2d_modules:
            mod.set_device(self.dev)

        for i_layer, kpt_affine in enumerate(l_kpt_pairs):
            if len(kpt_affine) == 2:
                kpt_F = kpt_affine[0].to(self.dev)
                kpt_M = kpt_affine[1].to(self.dev)
                if i_layer+1 >= len(self.grid2d_modules):
                    continue

                with torch.no_grad():
                    img_F, img_M = None, None
                    if show_img:
                        img_F = self.image_3d_tensor[i_layer:i_layer+1].to(self.dev)
                        img_M = self.image_3d_tensor[i_layer+1:i_layer+2].to(self.dev)

                    img_F_out, out_kpt_F = self.grid2d_modules[i_layer].forward(img_F, kpt_F)
                    img_M_out, out_kpt_M = self.grid2d_modules[i_layer+1].forward(img_M, kpt_M)
                    out_kpt_F = out_kpt_F * scale_level
                    out_kpt_M = out_kpt_M * scale_level
                    l_kpt_pairs_scaled.append([out_kpt_F.cpu(), out_kpt_M.cpu()])
                    if show_img:
                        l_imgs.append([img_F_out.cpu(), img_M_out.cpu()])

            else:
                with torch.no_grad():
                    img = None
                    if show_img:
                        img = self.image_3d_tensor[i_layer:i_layer+1].to(self.dev)

                    img_out, out_kpt = self.grid2d_modules[i_layer].forward(img, kpt_affine.to(self.dev))
                    out_kpt = out_kpt * scale_level
                    l_kpt_pairs_scaled.append([out_kpt.cpu()])
                    if show_img:
                        l_imgs.append([img_out.cpu()])

        return l_kpt_pairs_scaled, l_imgs


    def viz_all(self,
            image_3d_tensor: Tensor_image_NCHW,
            l_kpt_pairs: Tensor_l_kpt_pair=None,
            batch_size: int=8
        ) -> Tuple[Tensor_image_NCHW, Tensor_l_kpt_pair]:
        N = image_3d_tensor.shape[0]
        transformed_images_all = []
        l_kpts_moved_all = []

        for start_idx in tqdm(range(0, N, batch_size)):
            end_idx = min(start_idx + batch_size, N)
            temp_end_idx = min(end_idx+1, N) if l_kpt_pairs is not None else end_idx
            batch_images = image_3d_tensor[start_idx:temp_end_idx]
            batch_kpts = l_kpt_pairs[start_idx:temp_end_idx] if l_kpt_pairs is not None else None
            transformed_images, l_kpts_moved = self.viz(batch_images, batch_kpts, start_idx, temp_end_idx)
            transformed_images = transformed_images[:-1] if temp_end_idx > end_idx else transformed_images
            transformed_images_all.append(transformed_images)
            if l_kpts_moved is not None:
                for kpt_pair in l_kpts_moved:
                    l_kpts_moved_all.append(kpt_pair)

        transformed_images_all = torch.cat(transformed_images_all, dim=0)
        return transformed_images_all, l_kpts_moved_all

    def load_config(self, file_config):
        """load color yaml config from jhuanglab defination

        Args:
            file_config (FILE)

        Returns:
            dict: color dict
        """
        with open(file_config, 'r') as f:
            template_string = f.read()
            data = yaml.load(template_string, Loader=yaml.FullLoader)
            return data

    def _configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optim_name, optim_params = _dict_params(self.config, self.model_type, name_key="optimizer")

        if "learning_rates" in self.config_model and self.current_epoch < len(self.config_model["learning_rates"]):
            optim_params["lr"] = self.config_model["learning_rates"][self.current_epoch]

        optim_func = load_from_string(optim_name)
        l_params = [p for module in self.grid2d_modules for p in module.parameters()]
        optimizer = optim_func(l_params, **optim_params)

        sched_name, sched_params = _dict_params(self.config, self.model_type, name_key="scheduler")
        if sched_name == "":
            self.scheduler:torch.optim.lr_scheduler.LRScheduler = None
            self.optimizer:torch.optim.Optimizer = optimizer
            return {"optimizer": optimizer}

        sched_func = load_from_string(sched_name)
        scheduler = sched_func(optimizer, **sched_params)
        self.scheduler:torch.optim.lr_scheduler.LRScheduler = scheduler
        self.optimizer:torch.optim.Optimizer = optimizer
        return optimizer, scheduler


    def _prepare_grid_module(self, batch_images: Tensor_image_NCHW, batch_indices: List[int]):
        _ = self.update_tensor_size(image_3d_tensor=batch_images)
        for index in batch_indices:
            grid2d_module: Grid2DModelDual = self.grid2d_modules[index]
            if self.current_epoch > 0:
                kwargs = self.config_model
                if hasattr(grid2d_module, "displacement_field"):
                    kwargs["initial_displacement_field"] = grid2d_module.displacement_field
                    grid2d_module = self.grid_model(
                        init_trs=grid2d_module.tensor_trs,
                        current_epoch=self.current_epoch,
                        tensor_size=self.tensor_size,
                        **kwargs
                    )

            grid2d_module.unfreeze_layer()
            self.grid2d_modules[index] = grid2d_module

    def _prepare_device(self,
                batch_images: Tensor_image_NCHW=None,
                l_kpt_pairs: Tensor_l_kpt_pair=None,
                batch_indices: List[int]=None,
                use_pyramid: bool=False
        ) -> Tuple[Tensor_image_NCHW, Tensor_l_kpt_pair]:

        dev = self.dev
        if l_kpt_pairs is not None:
            for idx, kpt_pair in enumerate(l_kpt_pairs):
                for i_cmp, kpts in enumerate(kpt_pair):
                    kpt_F = kpts[0]
                    kpt_M = kpts[1]
                    if len(kpt_F.shape) == 3:
                        kpt_F = kpt_F.squeeze(0)
                    if len(kpt_M.shape) == 3:
                        kpt_M = kpt_M.squeeze(0)

                    l_kpt_pairs[idx][i_cmp][0] = kpt_F.to(dev)
                    l_kpt_pairs[idx][i_cmp][1] = kpt_M.to(dev)

        if batch_images is not None:
            batch_images = batch_images.to(dev)

        if use_pyramid:
            batch_images = self.target_pyramid[self.current_epoch][batch_indices.cpu()].to(dev)
            for i_grid in range(len(self.grid2d_modules)):
                self.grid2d_modules[i_grid].tensor_size = batch_images.shape[2:]

        return batch_images, l_kpt_pairs


    def _freeze_non_roi_layers(self, batch_indices: List[int]):
        self.grid2d_modules[batch_indices[-1]].freeze_layer(fwd=False, inv=(self.weight_image == 0))
        if batch_indices[0] > 0:
            l_ovlp_freezed = [batch_indices[idx] for idx in range(min(self.overlap, len(batch_indices)))]
            for index in l_ovlp_freezed:
                grid2d_module = self.grid2d_modules[index]
                grid2d_module.freeze_layer()

        for index in self.config_model["freezed_layers"]:
            self.grid2d_modules[index].freeze_layer()

        l_freeze_states = []
        for _, grid_module in enumerate(self.grid2d_modules):
            l_freeze_states.append([grid_module.displacement_field.requires_grad, grid_module.displacement_field.shape])


    def _write_tensorboard(self, dict_loss: Dict[str, torch.Tensor], batch_indices: torch.Tensor, beg: int, end: int):
        for loss_name, loss_value in dict_loss.items():
            self.writer.add_scalar(f'train/{loss_name}/{beg}_{end}', loss_value, self.i_iter_global[beg])

        if "trs_save_prefix" in self.config_model:
            l_trs_save_prefix = self.config_model["trs_save_prefix"]
            for index in batch_indices:
                grid2d_module = self.grid2d_modules[index]
                for i_rtk, prefix in enumerate(l_trs_save_prefix):
                    self.writer.add_scalar(f'train/{index}/{prefix}', grid2d_module.tensor_trs[i_rtk], self.i_iter_global[beg])

    def train_dataloader(self):
        dataset = OverlappedImageLayerDataset(self.image_3d_tensor, self.l_kpt_pairs, self.batch_size, self.overlap)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader

    def update_tensor_size(self, image_3d_tensor: Tensor_image_NCHW) -> Tensor_image_NCHW:
        H, W = image_3d_tensor.shape[2], image_3d_tensor.shape[3]
        if self.use_pyramid:
            elem_0 = self.target_pyramid[self.current_epoch]
            H, W = elem_0.shape[2], elem_0.shape[3]

        self.tensor_size = [H, W]
        # if "tensor_resize" in self.config_model:
        #     self.tensor_size = self.config_model["tensor_resize"]


        image_3d_tensor_resized = F.interpolate(image_3d_tensor, size=self.tensor_size, mode="bilinear", align_corners=False)
        return image_3d_tensor_resized


def _dict_params(config_obj, model_type, name_key="optimizer"):
    dict_obj = {}
    if name_key in config_obj:
        dict_obj = config_obj[name_key]

    if name_key in config_obj[model_type]:
        dict_obj = config_obj[model_type][name_key]

    optim_name, optimizer_params = "", {}
    for k,v in dict_obj.items():
        if k == f"{name_key}_name":
            optim_name = v
            continue

        optimizer_params[k] = v

    return optim_name, optimizer_params

def train_model(model: OmniAligner, num_epochs=1, l_kpts_eval: List[Tensor_kpts_N_xy|Tensor_kpt_pair]=None, scale_level=1.0):
    train_loader = model.train_dataloader()
    model.current_epoch = 0
    validate_epoch(model, l_kpt_pairs=l_kpts_eval, scale_level=scale_level, show_img=model.show_ovlp)
    for epoch in range(num_epochs):
        for batch in train_loader:
            _, _, indices = batch
            epoch_loss = model.train_epoch(batch)
            str_log = (f"layer {indices[0][0]}-{indices[0][-1]}, Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            if hasattr(model.grid2d_modules[0], "displacement_field"):
                str_log += f", shape:{model.grid2d_modules[0].displacement_field.shape}"
            print(str_log)

        model.current_epoch += 1
        if l_kpts_eval is not None:
            validate_epoch(model, l_kpt_pairs=l_kpts_eval, scale_level=scale_level, show_img=model.show_ovlp)

    model.writer.close()


def validate_epoch(model: OmniAligner, l_kpt_pairs: List[Tensor_kpts_N_xy|Tensor_kpt_pair], scale_level=1., show_img=False):
    if l_kpt_pairs is None:
        return None

    l_kpt_pairs_scaled, l_imgs = model.process_aligned_kpts(l_kpt_pairs, scale_level, show_img)
    TRE_pairwise, TRE_accumulated = benchmark_kpts(l_kpt_pairs_scaled)
    print(f"TRE_pairwise: {TRE_pairwise.mean():.2f}, {TRE_pairwise.max():.2f}, {TRE_pairwise.std():.2f}")
    print(f"TRE_accumulated: {TRE_accumulated.mean():.2f}, {TRE_accumulated.max():.2f}, {TRE_accumulated.std():.2f}")

    if show_img:
        model.writer.add_scalar("validate/TRE/mean", TRE_pairwise.mean(), model.current_epoch)
        model.writer.add_scalar("validate/TRE/max", TRE_pairwise.max(), model.current_epoch)
        model.writer.add_scalar("validate/TRE/std", TRE_pairwise.std(), model.current_epoch)
        model.writer.add_scalar("validate/aTRE/mean", TRE_accumulated.mean(), model.current_epoch)
        model.writer.add_scalar("validate/aTRE/max", TRE_accumulated.max(), model.current_epoch)
        model.writer.add_scalar("validate/aTRE/std", TRE_accumulated.std(), model.current_epoch)
        fig = plot_ovlp_kpts(l_imgs, l_kpt_pairs_scaled, scale_level=scale_level, figsize=model.full_figsize, n_cols=model.full_n_cols)
        model.writer.add_figure("validate/kpts_ovlp", fig, model.current_epoch)

    return l_kpt_pairs_scaled


def load_config(file_config):
    with open(file_config, 'r', encoding="utf-8") as f:
        template_string = f.read()
        data = yaml.load(template_string, Loader=yaml.FullLoader)
        return data




def test_whole():
    os.chdir("/cluster/home/bqhu_jh/projects/omni/src/omnialigner/align/models/")
    
    padded_tensors = torch.load("/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/v6/05.align3d/pdac/affine_tensor.pt")
    l_kpt_pairs = torch.load("/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/v6/05.align3d/pdac/affine_kpts_adddense.pt")
    config = load_config("config.yaml")
    model = OmniAligner(
        image_3d_tensor=padded_tensors,
        l_kpt_pairs=l_kpt_pairs,
        dict_config=config["align"],
        model_type="nonrigid",
        save_prefix="nonrigid",
        log_prefix="nonrigid",
    )
    train_model(model, num_epochs=config["align"]["nonrigid"]["used_levels"])
    




def test_3579():
    os.chdir("/cluster/home/bqhu_jh/projects/omni/src/omnialigner/align/models/")
    
    i_beg, i_end = 300, 500
    j_beg, j_end = 700, 900

    om_data = Omni3D(config_info="config.yaml")
    
    tensor_kpts = torch.load("./nonrigid/nonrigid_kpts.pt")
    dict_model = torch.load("./nonrigid/nonrigid_model.pt")

    crop_region = (i_beg / om_data.max_size, i_end / om_data.max_size, j_beg / om_data.max_size, j_end / om_data.max_size)
    tensor_transformed = apply_disp_to_HD(om_data, l_layers=list(range(8)), crop_region=crop_region, dict_model=dict_model)
    tensor_transformed = torch.cat(tensor_transformed, dim=0)
    # img_cropped = om.tl.tensor2im(tensor_transformed.permute(1, 0, 2, 3))
    h_, w_ = tensor_transformed.shape[2], tensor_transformed.shape[3]
    H_, W_ = om_data.l_scales[0] * om_data.max_size, om_data.l_scales[0] * om_data.max_size
    l_kpt_pairs_cropped = select_keypoints(tensor_kpts.copy(), crop_region=crop_region, scale=(H_ / h_, W_ / w_))


    config = load_config("config.yaml")
    padded_tensors = F.interpolate(tensor_transformed, size=(1024, 1024), mode="bilinear", align_corners=True)
    model = OmniAligner(
        image_3d_tensor=padded_tensors,
        l_kpt_pairs=l_kpt_pairs_cropped,
        dict_config=config["align"],
        model_type="nonrigid_sub",
        save_prefix="nonrigid_3579",
        log_prefix="nonrigid_3579",
    )
    train_model(model, num_epochs=config["align"]["nonrigid_sub"]["used_levels"])


if __name__ == "__main__":
    test_whole()
    # test_3579()
