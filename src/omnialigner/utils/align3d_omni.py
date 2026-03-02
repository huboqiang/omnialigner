import os
import sys
from typing import List
from collections import OrderedDict
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

import torch
import scanpy as sc
import numpy as np
from tqdm import tqdm
import yaml
import cv2
import torch.nn as nn
import torch.nn.functional as F

import omnialigner as om
from omnialigner.plotting.h5ad_viz import keypoints_gpd, gdf_shape_to_image
from omnialigner.align.models.loss import ncc_local
from omnialigner.align.models.aligner import OmniAligner, train_model
from omnialigner.utils.omni_aligner_tools import detect_best_pose_two_stages_v2, move_pose
from omnialigner.utils.alignment_utils import  get_sample_config

plt = om.pl.plt

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-255)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def tensor3d_cat_to_rgb(tensor, color_cells):
    max_category = max(color_cells.keys())
    color_map = np.zeros((max_category + 1, 3), dtype=np.uint8)
    for cat_id, hex_color in color_cells.items():
        color_map[cat_id] = hex_to_rgb(hex_color)


    N, _, H, W = tensor.shape
    tensor_rgb = torch.zeros((N, 3, H, W), dtype=torch.float32)

    for i in range(N):
        category_img = (tensor[i, 0].numpy()).astype(np.int32)
        category_img = np.clip(category_img, -1, max_category)
        rgb_img = color_map[category_img]
        tensor_rgb[i] = torch.from_numpy(rgb_img.transpose(2, 0, 1) / 255.0)

    return tensor_rgb

def generate_color_dict(adata: sc.AnnData, tag: str, cmap: List[str]=None):
    if cmap is None:
        cmap = sc.pl.palettes.default_102

    l_name_used = [ v for v in adata.obs[tag].value_counts().index ]
    adata.obs["cluster"] = [ int(l_name_used.index(v)) if v in l_name_used else -1 for v in adata.obs[tag] ]
    color_cells = {k+1: cmap[i % len(cmap)] for i, k in enumerate(range(len(l_name_used)))}
    color_cells[-1] = "#888888"
    color_dict = {"l_name_used": l_name_used, "color_cells": color_cells}
    return color_dict


def kpt_inside(kpt, h, w):
    idx_x = (kpt[:,0] >= 0) & (kpt[:,0] < w)
    idx_y = (kpt[:,1] >= 0) & (kpt[:,1] < h)
    idx = idx_x & idx_y
    return idx

def detect_keypoints(tensor_F: torch.Tensor, tensor_M: torch.Tensor, detector_xfeat: nn.Module, detector_roma: nn.Module, filter_keypoints:bool = True, sample:str="", **kwargs) -> om.kp.KeypointPairs:
    l_angles = None
    l_flips = None
    l_scales = None
    if len(sample) > 3 and sample.startswith("abca"):
        l_angles = [0]
        l_scales = [1]
        l_flips = [[1], [1]]
        
    with torch.no_grad():
        kd : om.kp.KeypointPairs = detect_best_pose_two_stages_v2(
            tensor0=tensor_F,
            tensor1=tensor_M,
            l_padsizes_dense=[[200, 200, 200, 200]],
            detector=detector_xfeat,
            detector_dense=detector_roma,
            l_angles=l_angles,
            l_flips=l_flips,
            l_scales=l_scales,
            filter_keypoints=filter_keypoints,
            func_score=lambda t0, t1: -ncc_local(t0, t1)
        )

    return kd


# ==================== OmniAligner Alignment Pipeline ====================

def step1_h5ad_to_png(adata: sc.AnnData, l_layers: List, l_name_used: List[str], color_cells: dict, 
                       w: int = 1280, h: int = 1280, radius: int = 8, out_root: str = "", sample: str = "", **kwargs):
    """Step 1: Convert h5ad to PNG tensors (category images)"""
    
    if os.path.exists(f"{out_root}/omnialigner/image_3d_category.pth"):
        tensor_stack = torch.load(f"{out_root}/omnialigner/image_3d_category.pth")
        tensor_ref = torch.load(f"{out_root}/omnialigner/image_3d_category_ref.pth")
        return tensor_stack, tensor_ref
    
    l_tensors = []
    l_tensors_ref = []
    # _, _, _, label_col, radius = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    for idx, layer in tqdm(enumerate(l_layers), desc="Step1: h5ad -> png"):
        adata_z = adata[adata.obs["i_layer"] == layer]
        l_clusters = [ int(l_name_used.index(v))+1 if v in l_name_used else -1 for v in adata_z.obs[label_col] ]
        l_colors = [ color_cells[i] for i in l_clusters ]

        # kpt = extract_raw_coords(adata_z, sample) * np.array([w, h])
        kpt = adata_z.obsm["spatial_3D"][:, 0:2]
        gdf = keypoints_gpd(kpt, key={"clusters": l_clusters}, color_key={"color": l_colors}, radius=radius[0])
        np_uint8 = gdf_shape_to_image(gdf, key="clusters", w=w, h=h)[:, :, np.newaxis]
        tensor = om.tl.im2tensor(np_uint8) * 255
        l_tensors.append(tensor)


        # kpt = extract_ref_coords(adata_z, sample) * np.array([w, h])
        kpt = adata_z.obsm["spatial_ref"][:, 0:2]
        gdf = keypoints_gpd(kpt, key={"clusters": l_clusters}, color_key={"color": l_colors}, radius=radius[1])
        np_uint8 = gdf_shape_to_image(gdf, key="clusters", w=w, h=h)[:, :, np.newaxis]
        tensor = om.tl.im2tensor(np_uint8) * 255
        l_tensors_ref.append(tensor)

    tensor_stack = torch.concat(l_tensors)
    tensor_ref = torch.concat(l_tensors_ref)
    os.makedirs(f"{out_root}/omnialigner/", exist_ok=True)
    torch.save(tensor_stack, f"{out_root}/omnialigner/image_3d_category.pth")
    torch.save(tensor_ref, f"{out_root}/omnialigner/image_3d_category_ref.pth")
    
    return tensor_stack, tensor_ref


def step2_detect_keypoints(tensor_rgb_raw: torch.Tensor, out_root: str, sample: str, filter_keypoints: bool = True, i_interval: int = 1, **kwargs):
    """Step 2: Detect keypoints between adjacent layers"""
    

    detector_roma:nn.Module = om.kp.init_detector("roma_dense")
    detector_xfeat:nn.Module = om.kp.init_detector("xfeat")
    os.makedirs(f"{out_root}/omnialigner/keypoints/roma_dense_color/figure/", exist_ok=True)
    for i_layer in tqdm(range(tensor_rgb_raw.shape[0]-i_interval), desc="Step2: png -> kpt"):
        for i_layer_next in range(i_layer+1, i_layer+i_interval+1):
            file_kd = f"{out_root}/omnialigner/keypoints/roma_dense_color/{i_layer}.pth"
            file_kd_img = f"{out_root}/omnialigner/keypoints/roma_dense_color/figure/{i_layer}.png"
            if (i_layer_next-i_layer) > 1:
                file_kd = f"{out_root}/omnialigner/keypoints/roma_dense_color/{i_layer}_{i_layer_next}.pth"
                file_kd_img = f"{out_root}/omnialigner/keypoints/roma_dense_color/figure/{i_layer}_{i_layer_next}.png"
            
            if os.path.exists(file_kd):
                continue

            tensor_F = tensor_rgb_raw[i_layer:i_layer+1]
            tensor_M = tensor_rgb_raw[i_layer+1:i_layer+2]
            
            kd : om.kp.KeypointPairs = detect_keypoints(tensor_F, tensor_M, filter_keypoints=filter_keypoints, sample=sample, detector_xfeat=detector_xfeat, detector_roma=detector_roma)
            fig = kd.plot_dataset()

            fig.savefig(file_kd_img)

            del kd.dataset["image_input"]
            del kd.dataset["image_label"]
            torch.save(kd.dataset, file_kd)
            plt.close(fig)


def step3_stack_poses(tensor_rgb_raw: torch.Tensor, out_root: str, sample: str, refine_pose: bool = True, **kwargs):
    """Step 3: Stack poses from keypoint matches"""
    if os.path.exists(f"{out_root}/omnialigner/tensor_poses.pth"):
        tensor_poses = torch.load(f"{out_root}/omnialigner/tensor_poses.pth")
        return tensor_poses

    from omnialigner.utils.field_transform import tfrs_to_grid_M, grid_M_to_tfrs    
    l_poses = [torch.FloatTensor([0, 0, 0, 0, 0, 1, 1])]
    grid_M_pose_prev = torch.eye(3)
    N_layers = tensor_rgb_raw.shape[0]


    for i_layer in tqdm(range(N_layers-1), desc="Step3: kpt -> stack"):
        file_kpts = f"{out_root}/omnialigner/keypoints/roma_dense_color/{i_layer}.pth"
        pt_res = torch.load(file_kpts)

        best_pose = pt_res.get("best_pose", torch.FloatTensor([0, 0, 0, 0, 0, 1, 1]))
        best_pose_ = best_pose.clone()

        grid_M_ = tfrs_to_grid_M(best_pose_).cpu()
        grid_M = torch.eye(3)
        grid_M[0:2, :] = grid_M_

        grid_M_pose = grid_M @ grid_M_pose_prev
        pose = grid_M_to_tfrs(grid_M_pose[0:2, :])
        l_poses.append(pose)
        grid_M_pose_prev = grid_M_pose.clone()

    
    tensor_poses = torch.stack(l_poses)
    if refine_pose:
        from omnialigner.align.stack_func import smooth_angles_short_range
        tensor_poses_refined = tensor_poses.clone()
        refine_pose_axes = kwargs.get("refine_pose_axes", [1, 2, 3, 4])
        for i_axis in refine_pose_axes:
           tensor_poses_refined[:, i_axis] = tensor_poses[:, i_axis] - torch.FloatTensor(smooth_angles_short_range(tensor_poses[:, i_axis], window=5))

        tensor_poses = tensor_poses_refined

    torch.save(tensor_poses, f"{out_root}/omnialigner/tensor_poses.pth")
    return tensor_poses


def step4_affine_alignment(image_3d_tensor: torch.Tensor, l_kpt_pairs: List, out_root: str, sample: str, 
                           config_info: dict, device: str = 'cuda', **kwargs):
    """Step 4: Affine alignment using OmniAligner"""
    if os.path.exists(f"{out_root}/omnialigner/affine_aligned_tensor.pt") and os.path.exists(f"{out_root}/omnialigner/affine_aligned_kpts.pt"):
        aligned_tensor = torch.load(f"{out_root}/omnialigner/affine_aligned_tensor.pt")
        l_kpts_moved = torch.load(f"{out_root}/omnialigner/affine_aligned_kpts.pt")
        return aligned_tensor, l_kpts_moved
    
    image_3d_tensor_ = F.interpolate(image_3d_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    
    model = OmniAligner(
        image_3d_tensor=image_3d_tensor_, 
        l_kpt_pairs=l_kpt_pairs, 
        log_prefix=f"{out_root}/logs", 
        dict_config=config_info["align"],
        model_type="affine",
        save_prefix=f"{sample}/affine",
        full_figsize=(15, 15),
        full_n_cols=(8)
    )
    train_model(model, num_epochs=1)
    
    dev = torch.device("cpu")
    model.image_3d_tensor = image_3d_tensor
    model.tensor_size = [image_3d_tensor.shape[2], image_3d_tensor.shape[3]]
    for mod in model.grid2d_modules:
        mod.set_device(dev)
        mod.tensor_size = [image_3d_tensor.shape[2], image_3d_tensor.shape[3]]

    model.eval()
    model.dev = dev
    model.n_keypoints = -1
    aligned_tensor, l_kpts_moved = model.viz_all(image_3d_tensor, l_kpt_pairs=l_kpt_pairs)

    dict_model = OrderedDict()
    for idx in range(model.N):
        dict_model[f"{idx}.tensor_trs"] = model.grid2d_modules[idx].tensor_trs

    os.makedirs(f"{out_root}/omnialigner/", exist_ok=True)
    torch.save(aligned_tensor, f"{out_root}/omnialigner/affine_aligned_tensor.pt")
    torch.save(l_kpts_moved, f"{out_root}/omnialigner/affine_aligned_kpts.pt")
    torch.save(dict_model, f"{out_root}/omnialigner/affine_model.pth")
    
    return aligned_tensor, l_kpts_moved


def step5_nonrigid_alignment(aligned_tensor: torch.Tensor, l_kpts_moved: List, out_root: str, sample: str, 
                              config_info: dict, **kwargs):
    """Step 5: Non-rigid alignment using OmniAligner"""
    if os.path.exists(f"{out_root}/omnialigner/nonrigid_aligned_tensor.pt") and os.path.exists(f"{out_root}/omnialigner/nonrigid_aligned_kpts.pt"):
        aligned_tensor = torch.load(f"{out_root}/omnialigner/nonrigid_aligned_tensor.pt")
        l_kpts_moved = torch.load(f"{out_root}/omnialigner/nonrigid_aligned_kpts.pt")
        return aligned_tensor, l_kpts_moved
    
    model = OmniAligner(
        image_3d_tensor=aligned_tensor, 
        l_kpt_pairs=l_kpts_moved, 
        log_prefix=f"{out_root}/logs", 
        dict_config=config_info["align"],
        model_type="nonrigid",
        save_prefix="nonrigid",
        full_figsize=(15, 15),
        full_n_cols=(8)
    )
    n_epochs = config_info["align"]["nonrigid"].get("used_levels", 1)
    train_model(model, num_epochs=n_epochs)
    
    aligned_tensor_nr, l_kpts_moved_ = model.viz_all(model.image_3d_tensor, model.l_kpt_pairs)
    aligned_tensor_nr = aligned_tensor_nr.detach().cpu()
    l_kpts_moved_nr = []
    for layer_kpts in l_kpts_moved_:
        layer_kpts_moved = []
        for kpt_pair in layer_kpts:
            kpt_F = kpt_pair[0].detach().cpu()
            kpt_M = kpt_pair[1].detach().cpu()
            layer_kpts_moved.append([kpt_F, kpt_M])
        l_kpts_moved_nr.append(layer_kpts_moved)

    dict_model = OrderedDict()
    for idx in range(model.N):
        dict_model[f"{idx}.displacement_field"] = model.grid2d_modules[idx].displacement_field

    os.makedirs(f"{out_root}/omnialigner/", exist_ok=True)
    torch.save(aligned_tensor_nr, f"{out_root}/omnialigner/nonrigid_aligned_tensor.pt")
    torch.save(l_kpts_moved_nr, f"{out_root}/omnialigner/nonrigid_aligned_kpts.pt")
    torch.save(dict_model, f"{out_root}/omnialigner/nonrigid_model.pth")
    
    return aligned_tensor_nr, l_kpts_moved_nr


def step6_apply_transform_to_h5ad(adata: sc.AnnData, l_layers: List, l_name_used: List[str], color_cells: dict,
                                   out_root: str, sample: str, w: int = 1280, h: int = 1280, **kwargs):
    """Step 6: Apply omnialigner transform to h5ad and save results"""
    if os.path.exists(f"{out_root}/omnialigner/adata_concat.h5ad"):
        adata_aligned = sc.read_h5ad(f"{out_root}/omnialigner/adata_concat.h5ad")
        return adata_aligned

    from omnialigner.align.models.grid_2d import TRSModuleDual, DeeperHistRegModule
    # label_col = "parcellation_structure" if sample.split("-")[0] == "abca" else "mapped_celltype"
    # _, _, _, label_col, radius = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")

    model_poses = torch.load(f"{out_root}/omnialigner/tensor_poses.pth")
    params_affine = torch.load(f"{out_root}/omnialigner/affine_model.pth", map_location="cpu")
    params_nonrigid = torch.load(f"{out_root}/omnialigner/nonrigid_model.pth", map_location="cpu")

    l_kpt_raw = []
    l_kpt_affine = []
    l_kpt_nonrigid = []

    l_img_raw = []
    l_img_nonrigid = []
    l_adatas = []
    
    for i_layer, layer in tqdm(enumerate(l_layers), desc="Step6: nonrigid -> h5ad"):
        adata_z = adata[adata.obs["i_layer"] == layer]
        l_clusters = [ int(l_name_used.index(v))+1 if v in l_name_used else -1 for v in adata_z.obs[label_col] ]
        l_colors = [ color_cells[i] for i in l_clusters ]

        kpt = torch.from_numpy(adata_z.obsm["spatial_3D"][:, 0:2]).float()
        _, kpt = move_pose(model_poses[i_layer], tensor=None, kpt=kpt, max_size=[w, h])
        cell_pos = kpt / torch.FloatTensor([w, h])
        
        model_layer_1 = TRSModuleDual(tensor_size=[h, w])
        model_layer_1.tensor_trs = params_affine[f"{i_layer}.tensor_trs"]
        model_layer_2 = DeeperHistRegModule(tensor_size=[h, w], disp_type="bspline", cp_spacing=(0.1, 0.1), splines_type="cubic", final_hw=[h, w])
        model_layer_2.set_device(torch.device("cpu"))
        model_layer_2.displacement_field = params_nonrigid[f"{i_layer}.displacement_field"]

        _, cell_pos_affine = model_layer_1.forward(None, cell_pos)
        _, cell_pos_nonrigid = model_layer_2.forward(None, cell_pos_affine)
        l_kpt_raw.append(cell_pos.detach().cpu().numpy())
        l_kpt_affine.append(cell_pos_affine.detach().cpu().numpy())
        l_kpt_nonrigid.append(cell_pos_nonrigid.detach().cpu().numpy())

        kpt = l_kpt_raw[i_layer] * np.array([h, w])
        gdf = keypoints_gpd(kpt, key={"clusters": l_clusters}, color_key={"color": l_colors}, radius=radius[0])
        np_uint8 = gdf_shape_to_image(gdf, key="clusters", w=w, h=h)[:, :, np.newaxis]
        tensor = om.tl.im2tensor(np_uint8) * 255.0
        l_img_raw.append(tensor)
        
        kpt = l_kpt_nonrigid[i_layer] * np.array([h, w])
        gdf = keypoints_gpd(kpt, key={"clusters": l_clusters}, color_key={"color": l_colors}, radius=radius[0])
        np_uint8 = gdf_shape_to_image(gdf, key="clusters", w=w, h=h)[:, :, np.newaxis]
        tensor = om.tl.im2tensor(np_uint8) * 255.0
        l_img_nonrigid.append(tensor)

        adata_z.obsm["align_spatial_stacked"] = l_kpt_raw[i_layer] * np.array([w, h])
        adata_z.obsm["align_spatial_affine"] = l_kpt_affine[i_layer] * np.array([w, h])
        adata_z.obsm["align_spatial_nonrigid"] = l_kpt_nonrigid[i_layer] * np.array([w, h])
        l_adatas.append(adata_z)

    image_3d_tensor = torch.concat(l_img_raw)
    image_3d_nonrigid = torch.concat(l_img_nonrigid)

    torch.save(image_3d_tensor, f"{out_root}/omnialigner/image_3d_category_stacked_check.pth")
    torch.save(image_3d_nonrigid, f"{out_root}/omnialigner/image_3d_category_nonrigid.pth")

    
    adata_concat = sc.concat(l_adatas)
    adata_concat.write_h5ad(f"{out_root}/omnialigner/adata_concat.h5ad")
    
    return adata_concat


def apply_stacked_pose(tensor_rgb_raw:torch.FloatTensor, tensor_poses: torch.FloatTensor, out_root: str, i_interval: int = 1):
    # Prepare data for alignment
    l_tensors = [tensor_rgb_raw[0:1]]
    l_kpt_pairs = []
    np_hw = np.array([tensor_rgb_raw.shape[2], tensor_rgb_raw.shape[3]])
    
    for i_layer_curr in tqdm(range(tensor_rgb_raw.shape[0]-i_interval), desc="Stack layers with poses"):
        ll_kpt_pairs = []
        for i_layer_next in range(i_layer_curr+1, i_layer_curr+i_interval+1):
            outdir = f"{out_root}/omnialigner/keypoints/roma_dense_color"
            file_kd = f"{outdir}/{i_layer_curr}.pth"
            if (i_layer_next-i_layer_curr) > 1:
                file_kd = f"{outdir}/{i_layer_curr}_{i_layer_next}.pth"

            datasets = torch.load(file_kd)
            idx = datasets["index_matches"]

            kpt_0, kpt_1 = datasets["test_label"][idx], datasets["test_input"][idx]
            tensor_moved, kpt1_moved = move_pose(
                tensor_tfrs_pose=tensor_poses[i_layer_curr+1],
                tensor=tensor_rgb_raw[i_layer_curr+1:i_layer_curr+2],
                kpt=kpt_1,
                constant_values=0,
                max_size=[tensor_rgb_raw.shape[2], tensor_rgb_raw.shape[3]],
                using_torch=True
            )
            kpt0_moved = kpt_0
            if i_layer_curr > 0:
                _, kpt0_moved = move_pose(
                    tensor_tfrs_pose=tensor_poses[i_layer_curr],
                    kpt=kpt_0,
                    constant_values=0,
                    max_size=[tensor_rgb_raw.shape[2], tensor_rgb_raw.shape[3]]
                )
                
            kpt_pair = [kpt0_moved / np_hw, kpt1_moved / np_hw]
            ll_kpt_pairs.append(kpt_pair)
            if i_layer_next-i_layer_curr == 1:
                l_tensors.append(tensor_moved)
        
        l_kpt_pairs.append(ll_kpt_pairs)

    image_3d_tensor = torch.concat(l_tensors, axis=0)
    print(f"Image tensor shape: {image_3d_tensor.shape}")
    torch.save(image_3d_tensor, f"{out_root}/omnialigner/image_3d_category_stacked.pth")
    return image_3d_tensor, l_kpt_pairs

def run_omnialigner_alignment(sample=None, device='cuda'):
    """
    Run OmniAligner alignment pipeline for 3D spatial data.
    
    Pipeline: h5ad -> png -> kpt -> stack -> affine -> nonrigid -> h5ad
    
    Args:
        sample: Sample name
        device: Device for computation ('cuda' or 'cpu')
    
    Returns:
        adata_concat: Concatenated AnnData with aligned coordinates
    """
    from alignment_utils import load_or_create_adata
    
    # sample, h5ad_path, out_root, label_col, radius = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    ref_layer = config_dict.get("ref_layer", 0)
    refine_pose = config_dict.get("refine_pose", True)
    refine_pose_axes = config_dict.get("refine_pose_axes", [1, 2, 3, 4])
    i_interval = config_dict.get("i_interval", 1)

    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running OmniAligner alignment for {sample}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    
    # Config
    om_config = "./config_align.yaml"
    w, h = 1280, 1280
    
    # Load config
    with open(om_config, 'r') as f:
        template_string = f.read()
        config_info = yaml.load(template_string, Loader=yaml.FullLoader)
        config_info['align']['affine']['lambda_L1_scale'] = 1000
        config_info['align']['affine']['iterations'] = [200, 200, 200]
        # config_info['align']['affine']['freezed_layers'] = [0, len(l_layers)-1]
        # config_info['align']['nonrigid']['freezed_layers'] = [0, len(l_layers)-1]
        # config_info['align']['nonrigid']['weight_reg'] = 1000.
    
    # Load h5ad
    print(f"Loading h5ad: {h5ad_path}")
    adata = load_or_create_adata(h5ad_path, out_root, sample, h=h, w=w, ref_layer=ref_layer)
    
    # Generate color dict
    color_dict = generate_color_dict(adata, tag=label_col, cmap=sc.pl.palettes.default_102)
    torch.save(color_dict, f"{out_root}/color_dict.pth")
    
    l_name_used = color_dict["l_name_used"]
    color_cells = color_dict["color_cells"]
    
    l_layers = adata.obs["i_layer"].cat.categories.tolist() if isinstance(adata.obs["i_layer"].dtype, pd.CategoricalDtype) else sorted(adata.obs["i_layer"].unique().tolist())
    
    # Update config with freezed layers

    
    # Step 1: h5ad -> png
    print("=" * 50)
    print("Step 1: h5ad -> png")
    tensor_stack, tensor_ref = step1_h5ad_to_png(
        adata, l_layers, l_name_used, color_cells, 
        w=w, h=h, radius=8, out_root=out_root, sample=sample
    )
    
    # Convert to RGB
    if os.path.exists(f"{out_root}/omnialigner/image_3d_category_rgb.pth"):
        tensor_rgb_raw = torch.load(f"{out_root}/omnialigner/image_3d_category_rgb.pth")
    else:
        tensor_rgb_raw = tensor3d_cat_to_rgb(tensor_stack, color_cells)
    
    
    # Step 2: png -> kpt
    print("=" * 50)
    print("Step 2: png -> kpt (detect keypoints)")
    step2_detect_keypoints(tensor_rgb_raw, out_root, sample, filter_keypoints=True, i_interval=i_interval)
    
    # Step 3: kpt -> stack
    print("=" * 50)
    print("Step 3: kpt -> stack (stack poses)")
    tensor_poses = step3_stack_poses(tensor_rgb_raw, out_root, sample, refine_pose=refine_pose, refine_pose_axes=refine_pose_axes)
    image_3d_tensor, l_kpt_pairs = apply_stacked_pose(tensor_rgb_raw, tensor_poses, out_root, i_interval=i_interval)

    # Step 4: stack -> affine
    print("=" * 50)
    print("Step 4: stack -> affine")
    aligned_tensor, l_kpts_moved = step4_affine_alignment(
        image_3d_tensor, l_kpt_pairs, out_root, sample, config_info, device
    )
    
    # Step 5: affine -> nonrigid
    print("=" * 50)
    print("Step 5: affine -> nonrigid")
    aligned_tensor_nr, l_kpts_moved_nr = step5_nonrigid_alignment(
        aligned_tensor, l_kpts_moved, out_root, sample, config_info
    )
    
    # Step 6: nonrigid -> h5ad
    print("=" * 50)
    print("Step 6: nonrigid -> h5ad")
    adata_concat = step6_apply_transform_to_h5ad(
        adata, l_layers, l_name_used, color_cells, out_root, sample, w, h
    )
    
    print("=" * 50)
    print(f"OmniAligner alignment completed!")
    print(f"Output saved to: {out_root}/omnialigner/adata_concat.h5ad")
    
    return adata_concat


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_omnialigner_alignment(sample=sys.argv[1])