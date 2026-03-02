import os
import sys
import subprocess
from typing import Tuple, List, Union
import torch
import torch.nn.functional as F
import dask.array as da
import json
import shapely
import numpy as np
import geopandas as gpd
import cv2
import heapq

from tqdm import tqdm

from omnialigner.align.models.loss import ncc_local
from omnialigner.plotting.h5ad_viz import gdf_shape_to_image
from omnialigner.utils.image_pad import center_pad_with_flank
from omnialigner.utils.image_transform import apply_tfrs_to_dask
from omnialigner.utils.field_transform import calculate_M_from_theta, tfrs_inv, grid_M_to_tfrs, tfrs_to_grid_M, grid_2d_to_disp_field
from omnialigner.utils.point_transform import transform_keypoints, warp_landmark_grid_faiss
import omnialigner as om

plt = om.pl.plt

def process_trident_segment(
        file_ome_v2,
        confidence_thresh: float=0.3,
        target_mag: int=20,
        patch_size: int=2560,
        result_dir: str=None,
        save_coords: str="output_coords",
        overlap: int = 320,
    ):
    import h5py
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory
    slide = load_wsi(slide_path=file_ome_v2, lazy_init=False)
    print("Running tissue segmentation...")
    segmentation_model = segmentation_model_factory(
        model_name="hest",
        confidence_thresh=confidence_thresh,
    )
    segmentation_model.precision = torch.float32
    result_dir = os.path.dirname(file_ome_v2) if result_dir is None else result_dir
    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=result_dir,
        device="cuda:0"
        # device=f"cpu"
    )
    file_coords = slide.extract_tissue_coords(target_mag, patch_size, save_coords, overlap=overlap)
    with h5py.File(file_coords, 'r') as f:
        attrs = dict(f['coords'].attrs)
        coords = f['coords'][:]

    return coords


def update_indices(current_indices, from_pos, to_pos):
    """
    Move element at position from_pos to position to_pos.
    
    Args:
        current_indices: List of indices
        from_pos: Source position (0-based)
        to_pos: Target position (0-based)
    
    Returns:
        updated_indices: New list after moving
    """
    num_layers = len(current_indices)
    assert 0 <= from_pos < num_layers, f"from_pos must be in [0, {num_layers-1}]"
    assert 0 <= to_pos < num_layers, f"to_pos must be in [0, {num_layers-1}]"
    
    if from_pos == to_pos:
        return current_indices.copy()
    
    updated_indices = current_indices.copy()
    element = updated_indices.pop(from_pos)  # 移除源位置的元素
    updated_indices.insert(to_pos, element)   # 插入到目标位置
    
    return updated_indices



def insert_layer(tensor_emb, i_layer, new_pos):
    """
    Move the i_layer-th layer in tensor_emb to new_pos, shifting other layers accordingly.
    (Note: Indices start from 0; new_pos refers to the target index after movement.)
    
    Args:
        tensor_emb: Tensor with shape [num_layers, ...] (first layer index = 0)
        i_layer: Original index of the layer to move (must be in [0, num_layers-1])
        new_pos: Target index after movement (must be in [0, num_layers-1])
    
    Returns:
        Adjusted tensor_emb (same length as the original tensor)
    """
    # Validate input indices to avoid out-of-bounds errors
    num_layers = tensor_emb.shape[0]
    assert 0 <= i_layer < num_layers, f"i_layer must be in [0, {num_layers-1}]"
    assert 0 <= new_pos < num_layers, f"new_pos must be in [0, {num_layers-1}]"
    
    # No operation needed if source and target positions are the same
    if i_layer == new_pos:
        return tensor_emb.clone()  # Return clone to maintain function purity
    
    # Extract the target layer (clone to avoid modifying the original tensor)
    target_layer = tensor_emb[i_layer:i_layer+1].clone()
    
    # Step 1: Remove the i_layer-th layer from the original tensor
    if i_layer < num_layers - 1:
        # Case 1: i_layer is not the last layer → concatenate parts before and after
        tensor_emb_without_target = torch.cat([
            tensor_emb[:i_layer],
            tensor_emb[i_layer+1:]
        ], dim=0)
    else:
        # Case 2: i_layer is the last layer → take all layers except the last
        tensor_emb_without_target = tensor_emb[:i_layer]
    
    # Step 2: Insert the target layer at new_pos
    if new_pos < tensor_emb_without_target.shape[0]:
        # Case 1: new_pos is not the last position → insert in between
        result = torch.cat([
            tensor_emb_without_target[:new_pos],
            target_layer,
            tensor_emb_without_target[new_pos:]
        ], dim=0)
    else:
        # Case 2: new_pos is the last position → append to the end
        result = torch.cat([tensor_emb_without_target, target_layer], dim=0)
    
    return result

def calculate_layer_pos_by_corr(tensor_emb, i_layer):
    l_corr = F.cosine_similarity(tensor_emb[i_layer:i_layer+1], tensor_emb, dim=3).mean((1,2))
    sim_orders = l_corr.argsort(descending=True)[1:]
    return sim_orders

def reorder_layers_by_corr(tensor_emb, shuffle_indices=None, iterations:int=1, top_k:int=1):
    if shuffle_indices is None:
        shuffle_indices = list(range(tensor_emb.shape[0]))

    for _ in tqdm(range(iterations)):
        for i_layer in range(tensor_emb.shape[0]):
            sim_orders = calculate_layer_pos_by_corr(tensor_emb[shuffle_indices, :, :, :], i_layer)
            best_pos = int(np.mean(sim_orders[0:top_k].numpy()))
            shuffle_indices = update_indices(shuffle_indices, i_layer, best_pos)
        
    return shuffle_indices


def raw_kpt_to_pad(np_kpts, file_qptiff, crop_size=None, ratio=1.0, pad_size=None, zoom_level:int=4, scale_factor:float=1.0, use_page:bool=False, **kwargs):
    """
    Transform cell keypoints from original coordinates to padded image coordinates.
    
    Transformation steps:
    1. Scale by pyramid level (2^zoom_level)
    2. Apply crop offset (if cropped)
    3. Scale by resize ratio
    4. Apply padding offset (left, top)
    """
    
    # Step 1: Scale by pyramid level
    np_kpts = np_kpts * scale_factor
    kwargs = {}
    if use_page:
        kwargs["i_page"] = zoom_level
    else:
        kwargs["i_level"] = zoom_level

    # Step 2: Apply crop offset
    if crop_size is not None:
        y1, x1, y2, x2 = crop_size
        da_img0 = om.tl.read_ome_tiff(file_qptiff, i_page=0, i_level=0)
        h, w = da_img0.shape[0], da_img0.shape[1]
    
        crop_offset = np.array([x1 * w, y1 * h])
        np_kpts = np_kpts - crop_offset
    
    # Step 3: Scale by resize ratio
    np_kpts = np_kpts / ratio
    
    # Step 4: Apply padding offset (left, top)
    if pad_size is not None:
        np_pad = np.array([pad_size[0], pad_size[2]])
        np_kpts = np_kpts + np_pad

    return np_kpts

def pad_kpt_to_raw(np_kpts, file_qptiff, crop_size=None, ratio=1.0, pad_size=None, zoom_level:int=4, scale_factor:float=1.0, use_page:bool=False, **kwargs):
    """
    Transform cell keypoints from padded image coordinates back to original coordinates.
    
    Transformation steps (reverse of raw_kpt_to_pad):
    4. Remove padding offset (reverse of step 4 in raw_kpt_to_pad)
    3. Scale by resize ratio (reverse of step 4)
    2. Apply crop offset (reverse of step 3)
    1. Scale by pyramid level (reverse of step 2)
    """
    
    # Step 4 (reverse): Remove padding offset (left, top)
    if pad_size is not None:
        np_pad = np.array([pad_size[0], pad_size[2]])
        np_kpts = np_kpts - np_pad

    # Step 3 (reverse): Scale by resize ratio
    np_kpts = np_kpts * ratio

    # Step 2 (reverse): Apply crop offset
    if crop_size is not None:
        y1, x1, y2, x2 = crop_size
        da_img0 = om.tl.read_ome_tiff(file_qptiff, i_page=0, i_level=0)
        h, w = da_img0.shape[0], da_img0.shape[1]
        crop_offset = np.array([x1 * w, y1 * h])
        np_kpts = np_kpts + crop_offset

    # Step 1 (reverse): Scale by pyramid level
    np_kpts = np_kpts / scale_factor
    return np_kpts


def pad_kpt_to_align(np_kpts, gridM, max_size=1280):
    coords_normalized = torch.from_numpy(np_kpts).float() / max_size
    coords_warped = warp_landmark_grid_faiss(
        coords_normalized,  # Add batch dimension
        grid=gridM[0:1]
    )[:, 0, :]  # Shape: [N_total, 2]
    coords_warped_denorm = coords_warped.numpy() * max_size
    return coords_warped_denorm



def apply_cell_movement(gdf1: gpd.GeoDataFrame, gridM: torch.FloatTensor, max_size=1280):
    gdf1_transformed = gdf1.copy()

    # Step 1: Extract all coordinates and track their indices
    all_coords = []
    coord_indices = []  # Track which geometry each coordinate belongs to
    coord_counts = []   # Track how many coordinates each geometry has

    for idx in tqdm(range(len(gdf1_transformed)), "Extracting coordinates"):
        geom = gdf1_transformed.loc[idx, 'geometry']
        
        if geom.geom_type == 'Polygon':
            coords = np.array(geom.exterior.coords[:-1])
        elif geom.geom_type == 'Point':
            coords = np.array([[geom.x, geom.y]])
        else:
            coord_counts.append(0)
            continue
        
        all_coords.append(coords)
        coord_indices.extend([idx] * len(coords))
        coord_counts.append(len(coords))

    # Step 2: Batch transform all coordinates at once
    if len(all_coords) > 0:
        all_coords_concat = np.vstack(all_coords)  # Shape: [N_total, 2]
        
        coords_warped_denorm = pad_kpt_to_align(all_coords_concat, gridM)
        
        # Step 3: Distribute transformed coordinates back to geometries
        start_idx = 0
        geom_idx = 0
        for idx in tqdm(range(len(gdf1_transformed)), "Transforming back geometries"):
            if coord_counts[idx] == 0:
                continue
                
            geom = gdf1_transformed.loc[idx, 'geometry']
            end_idx = start_idx + coord_counts[idx]
            coords_transformed = coords_warped_denorm[start_idx:end_idx]
            
            # Create new geometry
            if geom.geom_type == 'Polygon':
                gdf1_transformed.loc[idx, 'geometry'] = shapely.Polygon(coords_transformed)
            elif geom.geom_type == 'Point':
                gdf1_transformed.loc[idx, 'geometry'] = shapely.Point(coords_transformed[0])
            
            start_idx = end_idx


    return gdf1_transformed


def _mask_outside_contour(gdf: gpd.GeoDataFrame, da_img: da.Array|np.ndarray, fill_value=0):
    gdf["cluster"] = 1
    mask = gdf_shape_to_image(gdf, w=da_img.shape[1], h=da_img.shape[0])
    mask = np.repeat(mask[:, :, None], da_img.shape[2], axis=2)
    if isinstance(da_img, da.Array):
        da_img = da_img.compute()

    da_img[mask==0] = 255
    return da_img


def move_pose(tensor_tfrs_pose, tensor=None, kpt=None, tile_size=512, constant_values=255, max_size=None, **kwargs):
    if max_size is None and tensor is not None:
        max_size = [tensor.shape[2], tensor.shape[3]]

    if max_size is None:
        raise ValueError("Either tensor or max_size must be provided.")
    

    tensor_moved, kpt1_raw = None, None
    if tensor is not None:
        using_torch = kwargs.get('using_torch', False)
        if using_torch:
            grid = F.affine_grid( tfrs_to_grid_M(tensor_tfrs_pose).unsqueeze(0), size=(1, 1, tensor.shape[2], tensor.shape[3]) )
            tensor_moved = F.grid_sample(tensor, grid, mode='nearest', padding_mode='border', align_corners=True)
        else:
            da_image_pad = da.from_array(om.tl.tensor2im(tensor), chunks=(tile_size, tile_size, 3))
            da_image_moved = apply_tfrs_to_dask(da_image_pad, [tensor_tfrs_pose], tile_size=(tile_size, tile_size, 1), constant_values=constant_values)
            tensor_moved = om.tl.im2tensor(da_image_moved.compute())


    if kpt is not None:
        grid_M_1_inv = calculate_M_from_theta(tfrs_inv(tensor_tfrs_pose), h=max_size[0], w=max_size[1])[0:2]
        kpt1_raw = transform_keypoints(kpt, grid_M_1_inv)

    return tensor_moved, kpt1_raw

def calculate_pose_from_kpt(kpt0, kpt1, l_idxs, max_size=None):
    if max_size is None:
        max_size = np.array([1280, 1280])

    max_size = np.array(max_size) / 2.0
    np_kpt1, np_kpt0 = (kpt1[l_idxs].numpy() / max_size - 1), (kpt0[l_idxs].numpy() / max_size - 1)
    np_kpt1_homo = np.hstack([np_kpt1, np.ones((len(np_kpt1), 1))])  # [N, 3]
    np_affine_matrix = np.linalg.lstsq(np_kpt1_homo, np_kpt0, rcond=None)[0]  # [3, 2]

    np_affine_matrix_inv = np.linalg.inv(np.hstack([np_affine_matrix, np.array([[0], [0], [1]])]))[:, 0:2]
    best_pose = grid_M_to_tfrs(torch.from_numpy(np_affine_matrix_inv.T))
    return best_pose

def make_results(best_res, max_size=None):
    kpt0, kpt1, l_idxs, res, raw_pose, score = best_res
    optim_pose = calculate_pose_from_kpt(kpt0, kpt1, l_idxs, max_size=max_size)
    best_res = (kpt0, kpt1, l_idxs, res, optim_pose, score)
    return best_res


def detector_kpt_with_score(detector, tensor0, tensor1_moved):
    kpt0, kpt1, l_idxs, res = om.kp.match(
        tensor0,
        tensor1_moved,
        detector=detector
    )
    detector_name = detector.__module__.split(".")[-1]
    if len(detector_name)>=4 and detector_name[0:4] == "roma":
        kpt0 = kpt0.detach().cpu()
        kpt1 = kpt1.detach().cpu()
        np_masks = F.interpolate(tensor0, size=(128, 128))[0].mean(dim=0).cpu().numpy() < 1.0
        np_certainty = res[8]["certainty"].cpu().detach().numpy()[0, 0]
        np_certainty[np_masks==0] = np.nan
        score = np.nansum(np_certainty)
        return kpt0, kpt1, l_idxs, res, score

    return kpt0, kpt1, l_idxs, res, len(l_idxs)


def detect_with_pose(tensor0, tensor1, detector, tensor_tfrs_pose=None, max_size=None, score_func=None):
    if tensor_tfrs_pose is None:
        tensor_tfrs_pose = torch.tensor([np.deg2rad(0), 0, -0, 0, 0, 1.0, 1.0]).float()
    
    tensor1_moved, _ = move_pose(tensor_tfrs_pose, tensor=tensor1, max_size=max_size)
    kpt0, kpt1, l_idxs, res, score = detector_kpt_with_score(detector, tensor0, tensor1_moved)
    tensor_tfrs_pose_inv = grid_M_to_tfrs(tfrs_inv(tensor_tfrs_pose)[0:2])

    if max_size is None:
        max_size = tensor0.shape[2:4]

    _, kpt1_raw = move_pose(tensor_tfrs_pose_inv, kpt=kpt1, max_size=max_size)
    if score_func is not None:
        score = score_func(tensor0, tensor1_moved)

    return kpt0, kpt1_raw, tensor1_moved, l_idxs, res, score


def generate_poses(l_angles, l_scales=None, l_flips=None, l_padsizes=None):
    if l_angles is None:
        l_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    if l_scales is None:
        l_scales = [1.0]
    
    if l_flips is None:
        l_flips = [[1, -1], [1, -1]]

    if l_padsizes is None:
        l_padsizes = [None]

    l_poses = []
    for flip_x in l_flips[0]:
        for flip_y in l_flips[1]:
            if flip_x == -1 and flip_y == -1:
                continue

            for scale_ in l_scales:
                for angle in l_angles:
                    for pad_size in l_padsizes:
                        scale = -np.log(scale_)
                        tensor_tfrs_pose = torch.tensor([np.deg2rad(angle), 0, 0, scale, scale, flip_x, flip_y]).float()
                        l_poses.append([tensor_tfrs_pose, pad_size])
    return l_poses


# FUNC4: find best pose
def detect_best_pose(tensor0, tensor1, detector, l_angles=None, l_scales=None, l_flips=None, l_padsizes=None, max_size=None, optimize_pose_using_kpt=False, score_func=None):
    if l_angles is None:
        l_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    best_score = -np.inf
    best_res = None
    
    l_poses = generate_poses(l_angles, l_scales=l_scales, l_flips=l_flips, l_padsizes=l_padsizes)
    for pose in l_poses:
        tensor_tfrs_pose, pad_size = pose
        if pad_size is not None:
            pad_up, pad_down, pad_left, pad_right = pad_size
            tensor0 = F.pad(tensor0, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=1)
            tensor1 = F.pad(tensor1, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=1)

        kpt0, kpt1, tensor1_moved, l_idxs, res, score = detect_with_pose(tensor0, tensor1, detector, tensor_tfrs_pose, max_size=max_size, score_func=score_func)
        if pad_size is not None:
            tensor0 = tensor0[:, :, pad_up:-pad_down, pad_left:-pad_right]
            tensor1 = tensor1[:, :, pad_up:-pad_down, pad_left:-pad_right]
            kpt0 -= torch.FloatTensor([pad_left, pad_up])
            kpt1 -= torch.FloatTensor([pad_left, pad_up])

        if score > best_score:
            angle = np.rad2deg(tensor_tfrs_pose[0].item())
            flip_x, flip_y = tensor_tfrs_pose[5].item(), tensor_tfrs_pose[6].item()

            best_score = score
            best_res = (kpt0, kpt1, l_idxs, res, tensor_tfrs_pose, score)

    if best_res is None:
        return None
    
    if optimize_pose_using_kpt:
        max_size = [tensor0.shape[2], tensor0.shape[3]]
        best_res = make_results(best_res, max_size=max_size)

    return best_res

def detect_best_pose_two_stages(tensor0, tensor1, detector=None, detector_dense=None, l_padsizes=None, l_padsizes_dense=None, func_score=None):
    if detector is None:
        detector = om.kp.init_detector("xfeat")
    if detector_dense is None:
        detector_dense = om.kp.init_detector("roma_dense")

    kpt0, kpt1, l_idxs, res, best_pose, score = detect_best_pose(
        tensor0=tensor0,
        tensor1=tensor1,
        detector=detector,
        l_padsizes=l_padsizes,
        # l_angles=[0],
        # l_flips=[[1], [1]],
        # l_scales=[[1]],
        score_func=func_score
    )

    tensor1_moved, _ = move_pose(best_pose, tensor=tensor1, max_size=tensor0.shape[2:4])
    plt.imshow( om.tl.tensor2im(tensor1_moved) )


    kpt0, kpt1, l_idxs, res, best_pose2, score = detect_best_pose(
        tensor0=tensor0,
        tensor1=tensor1_moved,
        detector=detector_dense,
        l_padsizes=l_padsizes_dense,
        l_angles=[0],
        l_flips=[[1], [1]],
        l_scales=[[1]]
    )

    best_pose_inv = grid_M_to_tfrs(tfrs_inv(best_pose)[0:2])
    _, kpt1_rawpos = move_pose(best_pose_inv, kpt=kpt1, max_size=tensor0.shape[2:4])
    kd = om.kp.KeypointPairs(
        image_F=tensor0,
        image_M=tensor1,
        mkpts_F=kpt0,
        mkpts_M=kpt1_rawpos,
        index_matches=l_idxs,
    )
    kd.dataset["best_pose"] = best_pose
    kd.plot_dataset()
    return kd


def kpt_inside(kpt, h, w):
    """Check if keypoints are inside image bounds."""
    inside = (kpt[:, 0] >= 0) & (kpt[:, 0] < w) & (kpt[:, 1] >= 0) & (kpt[:, 1] < h)
    return inside.numpy() if isinstance(inside, torch.Tensor) else inside


def filter_keypoints_by_image(kpt0, kpt1, l_idxs, tensor0, tensor1, dilate_kernel=5, dilate_iterations=1):
    """
    Filter keypoints to keep only those inside valid (non-background) regions.
    """
    img_tar = om.tl.tensor2im(tensor0)
    img_src = om.tl.tensor2im(tensor1)
    
    img_tar_dilate = cv2.dilate(img_tar, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=dilate_iterations)
    img_src_dilate = cv2.dilate(img_src, np.ones((dilate_kernel, dilate_kernel), np.uint8), iterations=dilate_iterations)
    
    idx_inside0 = kpt_inside(kpt0, img_tar.shape[0], img_tar.shape[1])
    idx_inside1 = kpt_inside(kpt1, img_src.shape[0], img_src.shape[1])
    
    l_idxs_inside = np.array([idx for idx in l_idxs if idx_inside0[idx] and idx_inside1[idx]])
    
    if len(l_idxs_inside) == 0:
        return []
    
    idx0 = img_tar_dilate[kpt0[l_idxs_inside, 1].numpy().astype(np.int32), 
                          kpt0[l_idxs_inside, 0].numpy().astype(np.int32), 0] > 0
    idx1 = img_src_dilate[kpt1[l_idxs_inside, 1].numpy().astype(np.int32), 
                          kpt1[l_idxs_inside, 0].numpy().astype(np.int32), 0] > 0
    
    l_idxs_filtered = []
    for i_, idx in enumerate(l_idxs_inside):
        if idx0[i_] and idx1[i_]:
            l_idxs_filtered.append(idx)
    
    return l_idxs_filtered


def detect_best_pose_two_stages_v2(tensor0, tensor1, detector=None, detector_dense=None, 
                                    l_padsizes=None, l_padsizes_dense=None, func_score=None,
                                    l_angles=None, l_scales=None, l_flips=None,
                                    num_iterations=3, filter_keypoints=True, top_k=3, verbose=False, **kwargs
    ) -> om.kp.KeypointPairs:
    """
    Two-stage pose detection with iterative refinement using priority queue.
    
    Stage 1: Use sparse detector to find initial best poses from multiple candidates (heapq)
    Stage 2: Use dense detector with iterative refinement to optimize the pose
    
    Args:
        tensor0: Reference image tensor
        tensor1: Moving image tensor
        detector: Sparse keypoint detector (default: xfeat)
        detector_dense: Dense keypoint detector (default: roma_dense)
        l_padsizes: Padding sizes for stage 1
        l_padsizes_dense: Padding sizes for stage 2
        func_score: Custom scoring function
        l_angles: List of angles to search in stage 1
        l_scales: List of scales to search in stage 1
        l_flips: List of flip options for stage 1
        num_iterations: Number of refinement iterations in stage 2
        filter_keypoints: Whether to filter keypoints by valid image regions
        top_k: Number of top candidates from stage 1 to refine in stage 2
        verbose: Whether to print debug info
    
    Returns:
        kd: KeypointPairs object with best_pose
    """
    if detector is None:
        detector = om.kp.init_detector("xfeat")
    if detector_dense is None:
        detector_dense = om.kp.init_detector("roma_dense")

    max_size = [tensor0.shape[2], tensor0.shape[3]]

    # Stage 1: Find top-k best poses using sparse detector with heapq
    # Priority queue: (-score, unique_id, pose, pad_size, result)
    pq = []
    unique_id = 0  # To handle ties in heapq
    
    l_poses = generate_poses(l_angles, l_scales=l_scales, l_flips=l_flips, l_padsizes=l_padsizes)
    
    for pose_item in l_poses:
        tensor_tfrs_pose, pad_size = pose_item
        
        # Handle padding
        if pad_size is not None:
            pad_up, pad_down, pad_left, pad_right = pad_size
            tensor0_pad = F.pad(tensor0, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=1)
            tensor1_pad = F.pad(tensor1, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=1)
        else:
            tensor0_pad = tensor0
            tensor1_pad = tensor1

        kpt0, kpt1, tensor1_moved, l_idxs, res, score = detect_with_pose(
            tensor0_pad, tensor1_pad, detector, tensor_tfrs_pose, max_size=max_size, score_func=func_score
        )

        # Adjust keypoints for padding
        if pad_size is not None:
            kpt0 = kpt0 - torch.FloatTensor([pad_left, pad_up])
            kpt1 = kpt1 - torch.FloatTensor([pad_left, pad_up])

        # Push to priority queue (use negative score for max-heap behavior)
        heapq.heappush(pq, (-score, unique_id, tensor_tfrs_pose, pad_size, (kpt0, kpt1, l_idxs, res, tensor_tfrs_pose, score)))
        unique_id += 1

    if len(pq) == 0:
        return None

    if verbose:
        print(f"Stage 1 - Found {len(pq)} pose candidates")

    # Stage 2: Iterative refinement for top-k candidates
    best_kd = None
    best_final_score = -np.inf
    
    # Process top-k candidates from stage 1
    candidates_to_process = min(top_k, len(pq))
    
    for candidate_idx in range(candidates_to_process):
        if len(pq) == 0:
            break
            
        neg_score, _, init_pose, pad_size, stage1_res = heapq.heappop(pq)
        kpt0_s1, kpt1_s1, l_idxs_s1, res_s1, _, score_s1 = stage1_res
        
        if verbose:
            print(f"\nProcessing candidate {candidate_idx + 1}/{candidates_to_process} with stage1 score: {-neg_score}")

        # Iterative refinement using dense detector
        current_pose = init_pose.clone()
        best_kpt0, best_kpt1, best_l_idxs = None, None, None
        best_score = -np.inf
        best_pose_for_candidate = init_pose.clone()

        for iteration in range(num_iterations):
            # Apply current pose to tensor1
            tensor1_moved, _ = move_pose(current_pose, tensor=tensor1, max_size=max_size)
            
            if verbose:
                print(f"  Iteration {iteration + 1}/{num_iterations}")

            # Handle padding for dense detector
            pad_size_dense = None
            if l_padsizes_dense is not None and len(l_padsizes_dense) > 0:
                pad_size_dense = l_padsizes_dense[0]  # Use first padding size
            
            if pad_size_dense is not None:
                pad_up, pad_down, pad_left, pad_right = pad_size_dense
                tensor0_pad = F.pad(tensor0, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=1)
                tensor1_moved_pad = F.pad(tensor1_moved, pad=(pad_left, pad_right, pad_up, pad_down), mode='constant', value=1)
                # Calculate padded image size for pose computation
                max_size_padded = [tensor0_pad.shape[2], tensor0_pad.shape[3]]
            else:
                tensor0_pad = tensor0
                tensor1_moved_pad = tensor1_moved
                max_size_padded = max_size

            # Detect keypoints with dense detector (no rotation/flip search needed)
            # Use max_size_padded for detection since we're detecting on padded images
            kpt0, kpt1, _, l_idxs, res, score = detect_with_pose(
                tensor0_pad, tensor1_moved_pad, detector_dense, 
                tensor_tfrs_pose=torch.tensor([0., 0., 0., 0., 0., 1., 1.]).float(),
                max_size=max_size_padded, score_func=None
            )
            
            # Calculate score with custom function if provided
            if func_score is not None:
                score = func_score(tensor0, tensor1_moved)
            
            if len(l_idxs) == 0:
                if verbose:
                    print(f"    No matches found in iteration {iteration + 1}")
                break
            
            # Transform kpt1 back to original tensor1 coordinates
            # Note: kpt1 is in padded coordinate space, current_pose_inv should also use padded size
            current_pose_inv = grid_M_to_tfrs(tfrs_inv(current_pose)[0:2])
            
            # For inverse transform, we need to account for padding offset
            if pad_size_dense is not None:
                # First remove padding offset from kpt1 (kpt1 is in padded tensor1_moved space)
                kpt1_unpad = kpt1 - torch.FloatTensor([pad_left, pad_up])
                # Now kpt1_unpad is in unpadded tensor1_moved space, apply inverse pose
                _, kpt1_raw = move_pose(current_pose_inv, kpt=kpt1_unpad, max_size=max_size)
                
                # Also adjust kpt0 for padding
                kpt0_unpad = kpt0 - torch.FloatTensor([pad_left, pad_up])
            else:
                _, kpt1_raw = move_pose(current_pose_inv, kpt=kpt1, max_size=max_size)
                kpt0_unpad = kpt0
            
            # Filter keypoints if enabled (use unpadded coordinates)
            if filter_keypoints:
                l_idxs_filtered = filter_keypoints_by_image(kpt0_unpad, kpt1_raw, l_idxs, tensor0, tensor1)
                if verbose:
                    print(f"    Filtered keypoints: {len(l_idxs)} -> {len(l_idxs_filtered)}")
                l_idxs = l_idxs_filtered
            
            if len(l_idxs) < 3:
                if verbose:
                    print(f"    Not enough keypoints ({len(l_idxs)}) for pose optimization")
                break
            
            # Calculate optimized pose using unpadded coordinates and original max_size
            try:
                optimized_pose = calculate_pose_from_kpt(kpt0_unpad, kpt1_raw, l_idxs, max_size=max_size)
            except Exception as e:
                if verbose:
                    print(f"    Failed to calculate pose: {e}")
                break
            
            # Update best result for this candidate with the optimized pose
            if score > best_score or best_kpt0 is None:
                best_score = score
                best_kpt0 = kpt0_unpad  # Store unpadded coordinates
                best_kpt1 = kpt1_raw
                best_l_idxs = l_idxs
                best_pose_for_candidate = optimized_pose.float()
            
            if verbose:
                print(f"    Score: {score}, num_matches: {len(l_idxs)}")
            
            # Update current_pose for next iteration
            current_pose = optimized_pose.float()

        # Check if this candidate is the best overall
        if best_kpt0 is None:
            # Fallback to stage 1 results for this candidate
            best_kpt0 = kpt0_s1
            best_kpt1 = kpt1_s1
            best_l_idxs = l_idxs_s1
            best_score = score_s1
            best_pose_for_candidate = init_pose.clone()
        
        if best_score > best_final_score:
            best_final_score = best_score
            best_kd = om.kp.KeypointPairs(
                image_F=tensor0,
                image_M=tensor1,
                mkpts_F=best_kpt0,
                mkpts_M=best_kpt1,
                index_matches=best_l_idxs,
            )
            best_kd.dataset["best_pose"] = best_pose_for_candidate
            best_kd.dataset["best_score"] = best_final_score
            fig = best_kd.plot_dataset()
            # print(f"New best pose {candidate_idx}:{iteration}, {best_pose_for_candidate} score: {best_final_score}")
            # fig.savefig(f"best_pose_{candidate_idx}_{iteration}.png")
    
    if verbose and best_kd is not None:
        print(f"\nFinal best pose with {len(best_kd.dataset['index_matches'])} matches, score: {best_final_score}")
    
    return best_kd

def exec_trident(slide:str, dir_TRIDENT:str, model="conch_v15", mag=20, patch_size=256, seg_conf_thresh=0.5):
    exec_trident = "/cluster/home/bqhu_jh/projects/omni/src/omnialigner/vendor/TRIDENT/run_single_slide.py"
    os.makedirs(dir_TRIDENT, exist_ok=True)
    # Run TRIDENT
    cmds = ["python",
                    exec_trident,
                    "--slide_path", slide,
                    "--job_dir", dir_TRIDENT,
                    "--seg_conf_thresh", f"{seg_conf_thresh}",
                    "--patch_encoder", model,
                    "--mag", f"{mag}",
                    "--patch_size", f"{patch_size}",
                    "--attention"
    ]
    subprocess.run(cmds, check=True)


def exec_zs(slide: str, dir_zarr: str, overwrite_cache:bool=False):
    from wsidata import open_wsi
    import lazyslide as zs
    if not overwrite_cache and os.path.isdir(dir_zarr):
        wsi = open_wsi(slide, store=dir_zarr)
        return wsi
    
    wsi = open_wsi(slide, store=None)
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, 64)
    zs.seg.cells(wsi)
    import shutil
    if os.path.isdir(dir_zarr):
        shutil.rmtree(dir_zarr)

    wsi.write(dir_zarr, overwrite=overwrite_cache)
    return wsi


### FUNC1: step1_crop_trident
def step1_crop_trident(file_qptiff, file_geojson, ext_ratio=0.1, read_tiff_kwargs=None, **kwargs):
    if read_tiff_kwargs is None:
        read_tiff_kwargs = {}

    if not os.path.exists(file_geojson):
        # dir_TRIDENT = "/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/ANHIR/paired/trident_processed/COAD_01_5/S1/contours_geojson/S1.v2.ome.geojson"
        dir_TRIDENT = file_geojson.split("contours_geojson")[0]
        exec_trident(file_qptiff, dir_TRIDENT=dir_TRIDENT)

    with open(file_geojson, 'r') as f:
        geojson_contour = json.load(f)

    l_contours = geojson_contour["features"]
    if len(l_contours) == 0:
        return None

    gdf_coords = gpd.GeoDataFrame(geometry=[shapely.Polygon(contour['geometry']['coordinates'][0]) for contour in l_contours])
    coords = np.array([list(pt) for polygon in gdf_coords.geometry for pt in polygon.exterior.coords])
    min_x, min_y = coords.min(0)
    max_x, max_y = coords.max(0)
    h_, w_ = max_y - min_y, max_x - min_x
    da_img0 = om.tl.read_ome_tiff(file_qptiff, i_page=0, i_level=0)
    h, w = da_img0.shape[0], da_img0.shape[1]
    ext_h, ext_w = h_ * ext_ratio, w_ * ext_ratio
    x_beg, x_end = max(min_x - ext_w, 0), min(max_x + ext_w, w)
    y_beg, y_end = max(min_y - ext_h, 0), min(max_y + ext_h, h)

    x_beg_scaled, x_end_scaled, y_beg_scaled, y_end_scaled = x_beg / w, x_end / w, y_beg / h, y_end / h
    np_coords = np.array([y_beg_scaled, x_beg_scaled, y_end_scaled, x_end_scaled])

    da_img = om.tl.read_ome_tiff(file_qptiff, **read_tiff_kwargs)
    h_out, w_out = da_img.shape[0], da_img.shape[1]

    scale_h = h_out / h
    scale_w = w_out / w
    gdf_coords_scaled = gdf_coords.copy()
    gdf_coords_scaled.geometry = gdf_coords_scaled.geometry.translate(xoff=-x_beg, yoff=-y_beg)
    gdf_coords_scaled.geometry = gdf_coords_scaled.geometry.scale(xfact=scale_w, yfact=scale_h, origin=(0, 0))

    x_beg, x_end = int(x_beg_scaled * w_out), int(x_end_scaled * w_out)
    y_beg, y_end = int(y_beg_scaled * h_out), int(y_end_scaled * h_out)
    da_img = da_img[y_beg:y_end, x_beg:x_end, ...]
    da_img = _mask_outside_contour(gdf_coords_scaled, da_img)
    param = {"crop_size": np_coords, "da_img": da_img}

    return da_img, param


## FUNC2: pad
def step2_pad(da_img, max_size=None, pt_bg=None, **kwargs):
    if max_size is None:
        max_size = [1280, 1280]

    if isinstance(max_size, int):
        max_size = [max_size, max_size]

    padded_tensor, pad_size, ratio = center_pad_with_flank(om.tl.im2tensor(da_img), max_size=max_size, pt_bg=pt_bg)
    param = {"pad_size": pad_size, "ratio": ratio, "tensor": padded_tensor, "da_img": None}
    return padded_tensor, param


## FUNC3: cells
def step3_cell(file_cell_gpd, file_qptiff, crop_size=None, ratio=1.0, pad_size=None, read_tiff_kwargs=None, zoom_level=4, scale_factor:float=1.0, **kwargs):
    """
    Transform cell geometries from original coordinates to padded image coordinates.
    
    Transformation steps:
    1. Load cells with geometries
    2. Scale by pyramid level (2^zoom_level)
    3. Apply crop offset (if cropped)
    4. Scale by resize ratio
    5. Apply padding offset
    """
    if read_tiff_kwargs is None:
        read_tiff_kwargs = {}
    
    da_img = om.tl.read_ome_tiff(file_qptiff, **read_tiff_kwargs)
    if os.path.exists(file_cell_gpd):
        dir_zarr = file_cell_gpd.split("/shapes/")[0]
        exec_zs(file_qptiff, dir_zarr)
    
    gdf = gpd.read_parquet(file_cell_gpd)
    # Step 1: Scale by pyramid level
    # scale_factor = 1.0 / (2 ** zoom_level)
    gdf_transformed = gdf.copy()
    gdf_transformed.geometry = gdf_transformed.geometry.scale(
        xfact=scale_factor[0], yfact=scale_factor[1], origin=(0, 0)
    )
    
    # Step 2: Apply crop offset
    if crop_size is not None:
        y1, x1, y2, x2 = crop_size
        h, w = da_img.shape[0], da_img.shape[1]
        x_offset = x1 * w
        y_offset = y1 * h

        gdf_transformed.geometry = gdf_transformed.geometry.translate(
            xoff=-x_offset, yoff=-y_offset
        )
    
    # Step 3: Scale by resize ratio
    gdf_transformed.geometry = gdf_transformed.geometry.scale(
        xfact=1/ratio, yfact=1/ratio, origin=(0, 0)
    )
    
    # Step 4: Apply padding offset
    if pad_size is not None:
        pad_left, pad_right, pad_top, pad_bottom = pad_size
        gdf_transformed.geometry = gdf_transformed.geometry.translate(
            xoff=pad_left, yoff=pad_top
        )
    
    param = {"gdf_cell": gdf_transformed}
    return gdf_transformed, param


def load_grid_M(attr):
    grid_M = torch.eye(3).float()
    if attr["tensor_tfrs_pose"] is not None:
        pose = attr["tensor_tfrs_pose"].float()
        # print("Pose:", pose)
        pose[3] = 0.
        pose[4] = 0.
        grid_M[0:2] = tfrs_to_grid_M(pose)

    return grid_M



def plot_nchw_2d(
        om_data,
        l_layers = None,
        n_col = 10,
        n_row = 10,
        plt_figsize=(15, 15),
    ) -> plt.Figure:
    """Plot 2D images from an Omni3D dataset.

    Args:
        om_data: Omni3D dataset object.
        aligned_tag: Tag for aligned data (default: "PAD").
        l_layers: Optional list of layer indices to plot.
        l_kpt_pairs: Optional list of keypoint pairs to overlay.

    Returns:
        matplotlib.figure.Figure: The generated plot figure showing 2D slices.
    """
    from omnialigner.plotting.keypoint_viz import plot_kpt_pairs
    fig = plt.figure(figsize=plt_figsize)
    if l_layers is None:
        l_layers = om_data.l_layers
    
    grid = load_grid_M(om_data.dict_params[om_data.list_samples[0]])
    for idx,i_layer in enumerate(l_layers):
        ax = fig.add_subplot(n_col, n_row, idx+1)
        sample = om_data.list_samples[i_layer]
        attr = om_data.dict_params[sample]
        # tensor_pose = None
        # if idx > 0:
        #     grid_ = load_grid_M(om_data.dict_params[sample])
        #     grid = torch.matmul(grid_, grid)
        #     tensor_pose = grid_M_to_tfrs(grid[0:2])

        zoom_level = attr.get("zoom_level", -1)
        img = attr.viz_image(zoom_level=zoom_level).astype(np.uint8)
        l_kpt_pairs = None
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        kwargs = {
            "image": img,
            "title": f"{idx}:{sample.split('-')[-1]}",
            "ax": ax
        }
        if l_kpt_pairs is not None:
            if i_layer-1 >= 0 and i_layer-1 < len(om_data):
                kwargs["kpts0"] = l_kpt_pairs[i_layer-1][1]

            if i_layer >= 0 and i_layer < len(om_data):
                kwargs["kpts1"] = l_kpt_pairs[i_layer][0]

        plot_kpt_pairs(
            **kwargs
        )

    return fig

class AlignmentParams(object):
    def __init__(self, sample, 
                read_tiff_kwargs: dict = None,
                file_qptiff_template: str = '/cluster/home/bqhu_jh/projects/tp53/data/lxy_20251113/%s.svs',
                file_geojson_template: str = '/cluster/home/bqhu_jh/projects/tp53/analysis/tp53/v1/trident_processed/lxy_20251113/%s/contours_geojson/%s.geojson',
                file_cell_gpd_template: str = '/cluster/home/bqhu_jh/projects/tp53/analysis/tp53/v1/lazy_slide/lxy_20251113/%s_cellseg_plip.zarr/shapes/cells/shapes.parquet',
                file_cache_template: str = '/cluster/home/bqhu_jh/projects/tp53/analysis/tp53/v1/cache/lxy_20251113/pairwise_poses/%s.pth',
                max_height:int = 1280,
                max_width:int = 1280,
                max_size:int = 1280,
                pt_bg = torch.ones((1, 3, 1, 1)).float(),
                crop_size: np.ndarray = None,
                pad_size: np.ndarray = None,
                ratio: float = 1.0,
                tensor_tfrs_pose: torch.FloatTensor = None,
                grid_M: list = None,
                **kwargs
        ):
        """
        Initialize alignment parameters with file path templates.
        
        Args:
            sample: Sample name (e.g., "HE_10")
            read_tiff_kwargs: Dictionary of kwargs for reading TIFF files
            file_qptiff_template: Template for QPTIFF file path with %s placeholder
            file_geojson_template: Template for GeoJSON file path with %s placeholders (needs 2 for sample name)
            file_cell_gpd_template: Template for cell segmentation parquet file path with %s placeholder
            crop_size: Crop region [y1, x1, y2, x2] in normalized coordinates
            pad_size: Padding size [left, right, top, bottom]
            ratio: Resize ratio
            zoom_level: Pyramid zoom level
            tensor_tfrs_pose: Transformation pose tensor
            grid_M: List of transformation grids
        """
        self.sample = sample
        self.read_tiff_kwargs = read_tiff_kwargs if read_tiff_kwargs is not None else {}
        self.file_qptiff_template = file_qptiff_template
        self.file_geojson_template = file_geojson_template
        self.file_cell_gpd_template = file_cell_gpd_template
        self.file_cache_params = file_cache_template % self.sample
        self.get_file_paths()
        self.max_height = max_height
        self.max_width = max_width
        self.max_size = max_size
        self.pt_bg = pt_bg

        self.crop_size = crop_size
        self.pad_size = pad_size
        self.ratio = ratio
        self.zoom_level = max(self.read_tiff_kwargs.get("i_level", 0), self.read_tiff_kwargs.get("i_page", 0))
        self.tensor_tfrs_pose = tensor_tfrs_pose
        self.gdf_cell = None
        self.grid_M = grid_M if grid_M is not None else []
        self.use_page = self._if_using_page()

    
        _, scale_to_tar = self._get_scale_factors(zoom_level=self.zoom_level, use_page=self.use_page)
        # self.scale_factor = 1.0 / np.mean(scale_to_tar) #1.0 / (2 ** self.zoom_level)
        self.scale_factor = 1.0 / scale_to_tar

    def _if_using_page(self):
        use_page = False
        da_img = om.tl.read_ome_tiff(self.file_qptiff, i_level=0, i_page=0, l_channels=[0])
        try:
            da_img_scale0 = om.tl.read_ome_tiff(self.file_qptiff, i_level=1, i_page=0, l_channels=[0])
            if da_img.shape[0] == da_img_scale0.shape[0]*2:
                da_img_scale1 = om.tl.read_ome_tiff(self.file_qptiff, i_level=1, i_page=0, l_channels=[0])
                if da_img.shape[0] != da_img_scale1.shape[0]*2:
                    use_page = True
        except:
            use_page = True
        
        return use_page
        
    def __getitem__(self, key):
        """Support dictionary-style access: attr["ratio"]"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in AlignmentParams")
    
    def __setitem__(self, key, value):
        """Support dictionary-style assignment: attr["ratio"] = 1.5"""
        setattr(self, key, value)
        if key == "sample":
            self.get_file_paths()

    def __contains__(self, key):
        """Support 'in' operator: "ratio" in attr"""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """Dictionary-style get with default value"""
        return getattr(self, key, default)
    
    def update(self, other_dict):
        """Update multiple attributes from dictionary"""
        for key, value in other_dict.items():
            setattr(self, key, value)
            if key == "zoom_level":
                _, scale_to_tar = self._get_scale_factors(zoom_level=value, use_page=self.use_page)
                self.scale_factor = 1.0 / scale_to_tar
            if key == "read_tiff_kwargs":
                _, scale_to_tar = self._get_scale_factors(zoom_level=self.zoom_level, use_page=self.use_page)
                self.scale_factor = 1.0 / scale_to_tar
            if key == "sample":
                self.get_file_paths()

        return self
    
    def keys(self):
        """Return all attribute keys"""
        return [k for k in self.__dict__.keys() if not k.startswith('_')]
    
    def values(self):
        """Return all attribute values"""
        return [getattr(self, k) for k in self.keys()]
    
    def items(self):
        """Return key-value pairs"""
        return [(k, getattr(self, k)) for k in self.keys()]
    
    def to_dict(self):
        """Convert to dictionary"""
        dict_ = self.get_file_paths()
        for k in self.keys():
            dict_[k] = getattr(self, k)
        return dict_

    def get_file_paths(self):
        """Generate file paths for the current sample"""
        self.file_qptiff = self.file_qptiff_template % self.sample
        self.file_geojson = self.file_geojson_template % (self.sample, self.sample)
        self.file_cell_gpd = self.file_cell_gpd_template % self.sample
        return {
            'file_qptiff': self.file_qptiff_template % self.sample,
            'file_geojson': self.file_geojson_template % (self.sample, self.sample),
            'file_cell_gpd': self.file_cell_gpd_template % self.sample
        }
    
    def copy(self):
        """Create a deep copy of this object"""
        import copy
        return copy.deepcopy(self)
    
    def __repr__(self):
        """String representation"""
        attrs = ', '.join(f"{k}={repr(v)}" for k, v in self.items())
        return f"AlignmentParams({attrs})"
    
    def __str__(self):
        """Human-readable string representation"""
        lines = [f"AlignmentParams for sample '{self.sample}':"]
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                lines.append(f"  {k}: {type(v).__name__} {getattr(v, 'shape', 'N/A')}")
            elif isinstance(v, str) and len(v) > 50:
                lines.append(f"  {k}: {v[:47]}...")
            else:
                lines.append(f"  {k}: {v}")
        return '\n'.join(lines)


    def viz_cells(self, subsample=-1) -> gpd.GeoDataFrame:
        if self.gdf_cell is None and os.path.exists(self.file_cell_gpd):
            self.gdf_cell = gpd.read_parquet(self.file_cell_gpd)

        gdf_cell = self.gdf_cell.copy()
        if subsample > 0:
            gdf_cell = gdf_cell.sample(n=min(subsample, len(gdf_cell)), random_state=42).reset_index()

        if self.grid_M is None or len(self.grid_M) == 0:
            return gdf_cell
        
        
        for grid_M in self.grid_M:
            gdf_cell = apply_cell_movement(gdf_cell, grid_M)
        
        return gdf_cell

    def _get_scale_factors(self, zoom_level=0, use_page=False):
        kwargs = self.read_tiff_kwargs.copy()
        zoom_tag = "i_page" if use_page else "i_level"
        kwargs[zoom_tag] = zoom_level
        image_HD = om.tl.read_ome_tiff(self.file_qptiff, **kwargs)
        
        kwargs[zoom_tag] = self.read_tiff_kwargs.get(zoom_tag, 0)
        image_target = om.tl.read_ome_tiff(self.file_qptiff, **kwargs)
        
        kwargs[zoom_tag] = 0
        image_raw = om.tl.read_ome_tiff(self.file_qptiff, **kwargs)
        # print(f"image shapes of given zoom_level=={zoom_level}:", image_HD.shape)
        # print(f"image shapes of zoom_level==0:", image_raw.shape)
        # print(f"image shapes of read_tiff kwargs zoom_level=={self.read_tiff_kwargs.get(zoom_tag, 0)}:", image_target.shape)
        scale_to_pad = np.array([image_HD.shape[0]/image_target.shape[0], image_HD.shape[1]/image_target.shape[1]])
        scale_to_tar = np.array([image_raw.shape[0]/image_target.shape[0], image_raw.shape[1]/image_target.shape[1]])
        return scale_to_pad, scale_to_tar

    def viz_image(self, zoom_level=0, tensor_pose=None, **kwargs) -> da.Array:
        extend_name = self.file_qptiff.split(".")[-1]
        base, _ = os.path.splitext(self.file_qptiff)
        file_cached_large = f"{base}.aligned_zoom{zoom_level}.tiff"
        if zoom_level == 0 and os.path.exists(file_cached_large):
            print(file_cached_large)
            return om.tl.read_ome_tiff(file_cached_large, i_level=0, i_page=0, l_channels=[0,1,2])

        from omnialigner.align import apply_image_HD
        kwargs = self.read_tiff_kwargs.copy()
        zoom_tag = "i_page" if self.use_page else "i_level"
        kwargs[zoom_tag] = zoom_level
        image_HD = om.tl.read_ome_tiff(self.file_qptiff, **kwargs)

        scale_to_pad, _ = self._get_scale_factors(zoom_level=zoom_level, use_page=self.use_page)
        affine_matrix = np.zeros([3, 3])
        affine_matrix[0, 0] = self.ratio
        affine_matrix[1, 1] = self.ratio

        l_disps = None if len(self.grid_M) < 2 else [grid_2d_to_disp_field(grid_M) for grid_M in self.grid_M[1:]]
        pad_size = [self.pad_size[0]*scale_to_pad[0], self.pad_size[1]*scale_to_pad[1], self.pad_size[2]*scale_to_pad[0], self.pad_size[3]*scale_to_pad[1]] if self.pad_size is not None else None

        tensor_pose_used = self.tensor_tfrs_pose
        if tensor_pose is not None:
            tensor_pose_used = tensor_pose

        da_img = apply_image_HD(
            image_HD,
            crop_size=self.crop_size,
            pad_size=pad_size,
            affine_matrix=affine_matrix,
            tensor_tfrs_1=tensor_pose_used,
            l_disps=l_disps,
            **{"constant_values": int(255 * self.pt_bg.mean().item())}
        )
        if zoom_level == 0:
            om.tl.write_qptiff_2d(file_cached_large, da_img)

        return da_img
    
    def viz_keypoint(self, kpt, using_HD=False, is_raw=True, zoom_level=0, out_zoom_level=None, out_is_raw:bool=False, exteral_object=None, **kwargs):

        func_pad_kpt_to_align = pad_kpt_to_align_HD if using_HD else pad_kpt_to_align
        if is_raw:
            kpt0 = raw_kpt_to_pad(kpt, self.file_qptiff, crop_size=self.crop_size, ratio=self.ratio, pad_size=self.pad_size, zoom_level=zoom_level, use_page=self.use_page, scale_factor=self.scale_factor)
            kpt = kpt0.copy()
        
        i_grid_M = 0
        if self.tensor_tfrs_pose is not None:
            grid_M = F.affine_grid(tfrs_to_grid_M(self.tensor_tfrs_pose).unsqueeze(0), size=[1, 2, self.max_size, self.max_size])
            kpt0 = func_pad_kpt_to_align(kpt, grid_M, self.max_size)
            kpt = kpt0.copy()
            i_grid_M = 1
            # if self.grid_M is not None and len(self.grid_M) >= 2:
            #     for grid_M in self.grid_M[1:]:
            #         kpt = func_pad_kpt_to_align(kpt, grid_M, self.max_size)
        
        # else:
        if self.grid_M is not None and len(self.grid_M) >= 1:
            for grid_M in self.grid_M[i_grid_M:]:
                kpt = func_pad_kpt_to_align(kpt, grid_M, self.max_size)


        if out_is_raw:
            if exteral_object is None:
                exteral_object = self

            out_zoom_level = zoom_level if out_zoom_level is None else out_zoom_level
            _, scale_to_pad = exteral_object._get_scale_factors(zoom_level=out_zoom_level, use_page=exteral_object.use_page)
            scale_factor = 1.0 / scale_to_pad
            kpt = pad_kpt_to_raw(kpt, exteral_object.file_qptiff, crop_size=exteral_object.crop_size, ratio=exteral_object.ratio, pad_size=exteral_object.pad_size, zoom_level=out_zoom_level, use_page=exteral_object.use_page, scale_factor=scale_factor)
            return kpt
    
            if out_zoom_level is not None:
                _, scale_to_pad = self._get_scale_factors(zoom_level=out_zoom_level, use_page=self.use_page)
                kpt = kpt * scale_to_pad
                return kpt
        
        return kpt


    def viz_mask(self, zoom_level=0, **kwargs) -> da.Array:
        from omnialigner.align import apply_image_HD
        kwargs_tiff = self.read_tiff_kwargs.copy()
        zoom_tag = "i_page" if self.use_page else "i_level"
        kwargs_tiff[zoom_tag] = zoom_level
        image_HD = om.tl.read_ome_tiff(self.file_qptiff, **kwargs_tiff)
        
        kwargs_tiff[zoom_tag] = self.read_tiff_kwargs[zoom_tag]
        image_target = om.tl.read_ome_tiff(self.file_qptiff, **kwargs_tiff)
        
        kwargs_tiff[zoom_tag] = 0
        image_raw = om.tl.read_ome_tiff(self.file_qptiff, **kwargs_tiff)
        
        scale_to_pad = np.array([image_HD.shape[0]/image_target.shape[0], image_HD.shape[1]/image_target.shape[1]])
        scale_to_tar = np.array([image_raw.shape[0]/image_target.shape[0], image_raw.shape[1]/image_target.shape[1]])

        affine_matrix = np.zeros([3, 3])
        affine_matrix[0, 0] = self.ratio
        affine_matrix[1, 1] = self.ratio

        # l_disps = None if len(self.grid_M) < 2 else [grid_2d_to_disp_field(grid_M) for grid_M in self.grid_M[1:]]
        l_disps = None if len(self.grid_M) < 1 else [grid_2d_to_disp_field(grid_M) for grid_M in self.grid_M]
        pad_size = [self.pad_size[0]*scale_to_pad[0], self.pad_size[1]*scale_to_pad[1], self.pad_size[2]*scale_to_pad[0], self.pad_size[3]*scale_to_pad[1]] if self.pad_size is not None else None

        with open(self.file_geojson, 'r') as f:
            geojson_contour = json.load(f)

        l_contours = geojson_contour["features"]
        gdf_coords = gpd.GeoDataFrame(geometry=[shapely.Polygon(contour['geometry']['coordinates'][0]) for contour in l_contours])
        gdf_coords.geometry = gdf_coords.geometry.scale(xfact=1/scale_to_tar[1], yfact=1/scale_to_tar[0], origin=(0, 0))

        gdf_coords["cluster"] = 1
        mask = gdf_shape_to_image(gdf_coords, w=image_HD.shape[1], h=image_HD.shape[0])
        mask = np.repeat(mask[:, :, None], image_HD.shape[2], axis=2)
        return apply_image_HD(
            mask,
            crop_size=self.crop_size,
            pad_size=pad_size,
            affine_matrix=affine_matrix,
            tensor_tfrs_1=self.tensor_tfrs_pose,
            l_disps=l_disps,
            **{"constant_values": int(0 * self.pt_bg.mean().item())}
        )
    
    def save_cache_layer(self):
        file_cache = self.file_cache_params
        dirname = os.path.dirname(file_cache)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.to_dict(), file_cache)
    
    def load_cache_layer(self):
        file_cache = self.file_cache_params
        if os.path.exists(file_cache):
            param = torch.load(file_cache, weights_only=False)
            self.update(param)
            return True
        
        return False
    
    def update_cache_layer(self, key, value):
        file_cache = self.file_cache_params
        if os.path.exists(file_cache):
            param = torch.load(file_cache)
            param[key] = value
            torch.save(param, file_cache)
            setattr(self, key, value)
            return True
        
        return False
    
class SerialWSIs(object):
    def __init__(self,
            list_samples: List[str],
            l_layers: List[int]=None,
            file_pairwise_pose_template: str = '/cluster/home/bqhu_jh/projects/tp53/analysis/tp53/v1/04.detect_kpts/lxy_20251113/pairwise_poses/%s_to_%s_pose.pt',
            
            **kwargs
        ):
        self.list_samples = list_samples
        if l_layers is None:
            l_layers = list(range(len(list_samples)))

        self.l_layers = l_layers
        self.dict_params: dict[str, AlignmentParams] = {}
        for sample in list_samples:
            self.dict_params[sample] = AlignmentParams(sample, **kwargs)
        
        self.poses = [None for _ in range(len(list_samples))]
        self.file_pairwise_pose_template = file_pairwise_pose_template
    
    # def detect_pose(self, 
    #             i_layer: int,
    #             detector=None,
    #             detector_dense=None,
    #             l_angles: List[float]=None,
    #             l_scales: List[float]=None,
    #             l_flips: Tuple[Tuple[int, int],Tuple[int, int]]=None,
    #             l_padsizes: List[Tuple[int, int, int, int]]=None,
    #             override=False,
    #             showfig=False
    #     ) -> Tuple[torch.FloatTensor, plt.Figure]:
    #     """
    #     Detect the best pose between two layers using keypoint matching.

    #     Args:
    #         i_layer: Index of the current layer to align (aligns layer i_layer-1 to i_layer)
    #         detector: Keypoint detector object (if None, defaults to "xfeat")
    #         l_angles: List of rotation angles to consider (in degrees)
    #         l_scales: List of scales to consider
    #         l_flips: Tuple of flip options for x and y axes
    #         pad_size: Padding size as (up, down, left, right)
    #         optimize_pose_using_kpt: Whether to optimize the pose using keypoints after initial detection
    #         override: Whether to override existing cached results
    #         showfig: Whether to generate and save match visualization figures
        
    #     Returns:
    #         best_pose: Detected best transformation pose as a torch.FloatTensor
    #         fig: Matplotlib figure of keypoint matches (if showfig is True)
    #     """
        # if detector is None:
        #     detector = om.kp.init_detector("xfeat")
        # if detector_dense is None:
        #     detector_dense = om.kp.init_detector("roma_dense")

        # l_padsizes = l_padsizes if l_padsizes is not None else [[200, 200, 200, 200]]
        # sample0 = self.list_samples[i_layer-1]
        # sample1 = self.list_samples[i_layer]
        # fig = None
        # if i_layer >= len(self.list_samples) or i_layer <= 0:
        #     return None, fig
        
        # file_pairwise_pose = self.file_pairwise_pose_template % (sample0, sample1)
        # if os.path.exists(file_pairwise_pose) and not override:
        #     kd = om.kp.KeypointPairs()
        #     kd.dataset = torch.load(file_pairwise_pose)
        #     best_pose = kd.dataset["best_pose"]
        #     self.poses[i_layer] = best_pose
        #     h, w = self.dict_params[sample1].tensor.shape[2], self.dict_params[sample1].tensor.shape[3]
        #     self.dict_params[sample1].update({
        #         "tensor_tfrs_pose": best_pose,
        #         # "grid_M" : [F.affine_grid(tfrs_to_grid_M(best_pose).unsqueeze(0), size=[1, 2, h, w])]
        #     })
        #     if showfig:
        #         kd.dataset['image_input'] = self.dict_params[sample1].tensor
        #         kd.dataset['image_label'] = self.dict_params[sample0].tensor
        #         fig = kd.plot_dataset()
        #         fig.savefig(file_pairwise_pose.replace('_pose.pt', '_matches.png'), dpi=300)

        #     return best_pose, fig

        # tensor0 = self.dict_params[sample0].tensor
        # tensor1 = self.dict_params[sample1].tensor

        # kd = detect_best_pose_two_stages_v2(
        #     tensor0=tensor0,
        #     tensor1=tensor1,
        #     l_angles=l_angles,
        #     l_scales=l_scales,
        #     l_flips=l_flips,
        #     l_padsizes=l_padsizes,
        #     detector=detector,
        #     detector_dense=detector_dense,
        #     func_score=lambda t0, t1: -ncc_local(t0, t1)
        # )
        # best_pose = kd.dataset['best_pose']
        # self.poses[i_layer] = best_pose
        # dirname = os.path.dirname(file_pairwise_pose)
        # os.makedirs(dirname, exist_ok=True)
        # if showfig:
        #     fig = kd.plot_dataset()
        #     fig.savefig(file_pairwise_pose.replace('_pose.pt', '_matches.png'), dpi=300)

        # kd.dataset['image_input'] = None
        # kd.dataset['image_label'] = None
        # torch.save(kd.dataset, file_pairwise_pose)
        # h, w = self.dict_params[sample1].tensor.shape[2], self.dict_params[sample1].tensor.shape[3]
        # # grid_pose = F.affine_grid(tfrs_to_grid_M(best_pose).unsqueeze(0), size=[1, 2, h, w])
        # # l_grid_poses = [grid_pose]

        # self.dict_params[sample1].update({
        #     "tensor_tfrs_pose": best_pose,
        #     # "grid_M" : l_grid_poses
        # })
        # return best_pose, fig
    
    def concat_tensors(self):
        l_tensors = []

        for i_layer in self.l_layers:
            sample = self.list_samples[i_layer]
            attr = self.dict_params[sample]
            da_img = attr.viz_image(zoom_level=attr.zoom_level)
            l_tensors.append(om.tl.im2tensor(da_img))

        return torch.concat(l_tensors, dim=0)

    def tensors_embeddings(self, encoder_name="conch_v15", device='cpu'):
        from trident.patch_encoder_models.loader_attr import attr_encoder_factory 
        from PIL import Image
        
        device = torch.device(device) if isinstance(device, "str") else device
        encoder = attr_encoder_factory(encoder_name)
        tensor = self.tensor
        l_outputs = []
        for idx in range(tensor.shape[0]):
            with torch.no_grad():
                dummy_input = encoder.eval_transforms(Image.fromarray(om.tl.tensor2im(tensor[idx:idx+1]))).to(device).unsqueeze(dim=0)
                output = encoder(dummy_input)
                l_outputs.append(output)

        tensor_emb = torch.concat(l_outputs, dim=0)
        return tensor_emb


    def sort_layers(self, encoder_name="conch_v15", device='cpu'):
        ## concat NCHW tensor
        if self.tensor is None:
            self.tensor = self.concat_tensors()


        if self.tensor_emb is None:
            self.tensor_emb = self.tensors_embeddings(encoder_name=encoder_name, device=device)

        
        sim_orders = reorder_layers_by_corr(self.tensor_emb, iterations=5, top_k=5)
        return sim_orders

    def align_layers(self,
            out_root:str,
            sample="",
            om_config:str="/cluster/home/bqhu_jh/projects/omnialigner/src/omnialigner/utils/config_align.yaml",
            refine_pose:bool=True,
            refine_pose_axes:List=None,
            i_interval:int=1
        ):
        import yaml
        from omnialigner.align.models.grid_2d import DeeperHistRegModule
        from omnialigner.utils.align3d_omni import step2_detect_keypoints, step3_stack_poses, step4_affine_alignment, step5_nonrigid_alignment, apply_stacked_pose
        from omnialigner.utils.field_transform import F, tfrs_to_grid_M, grid_M_to_tfrs, disp_field_to_grid_2d


        if refine_pose_axes is None:
            refine_pose_axes = [0, 1, 2, 3]

        with open(om_config, 'r') as f:
            template_string = f.read()
            config_info = yaml.load(template_string, Loader=yaml.FullLoader)
            config_info['align']['affine']['lambda_L1_scale'] = 1000
            config_info['align']['affine']['iterations'] = [200, 200, 200]

        print("=" * 50)
        print("Step 1: crop and pad into same size, raw image -> 3D NCHW tensor")
        for i_layer, sample in enumerate(self.list_samples):
            attr0 = self.dict_params[sample]
            da_img0, param = step1_crop_trident(**attr0.to_dict())
            attr0.update(param)
            tensor0, param0 = step2_pad(**attr0)
            attr0.update(param0)

        tensor_rgb_raw = torch.concat([ self.dict_params[sample].tensor for sample in self.list_samples ], axis=0)
        
        print("=" * 50)
        print("Step 2: pairwise detect keypoints")
        step2_detect_keypoints(tensor_rgb_raw, out_root, sample=sample, filter_keypoints=True)

        # Step 3: kpt -> stack
        print("=" * 50)
        print("Step 3: kpt -> stack (stack poses)")
        tensor_poses = step3_stack_poses(tensor_rgb_raw, out_root, sample, refine_pose=refine_pose, refine_pose_axes=refine_pose_axes)
        image_3d_tensor, l_kpt_pairs = apply_stacked_pose(tensor_rgb_raw, tensor_poses, out_root, i_interval=i_interval)

        # Step 4: stack -> affine
        print("=" * 50)
        print("Step 4: stack -> affine")
        aligned_tensor, l_kpts_moved = step4_affine_alignment(
            image_3d_tensor, l_kpt_pairs, out_root, sample, config_info
        )

        # Step 5: affine -> nonrigid
        print("=" * 50)
        print("Step 5: affine -> nonrigid")
        aligned_tensor_nr, l_kpts_moved_nr = step5_nonrigid_alignment(
            aligned_tensor, l_kpts_moved, out_root, sample, config_info
        )


        self.file_cache_params = f"{out_root}/omnialigner/cache_params/%s.pth"
        dict_affine_model = torch.load(f"{out_root}/omnialigner/affine_model.pth")
        dict_nonrigid_model = torch.load(f"{out_root}/omnialigner/nonrigid_model.pth")
        for i_layer, sample in enumerate(self.list_samples):
            attr0 = self.dict_params[sample]
            h, w = attr0.tensor.shape[2], attr0.tensor.shape[3]
            best_pose = tensor_poses[i_layer]
            grid_M_ = tfrs_to_grid_M(best_pose).cpu()
            grid_M_pose = torch.eye(3)
            grid_M_pose[0:2, :] = grid_M_
            affine_pose = dict_affine_model[f"{i_layer}.tensor_trs"]
            grid_M_ = tfrs_to_grid_M(affine_pose).cpu()
            grid_M_affine = torch.eye(3)
            grid_M_affine[0:2, :] = grid_M_
            grid_M_combine = grid_M_affine @ grid_M_pose
            combined_pose = grid_M_to_tfrs(grid_M_combine[0:2, :])
            grid_combined = F.affine_grid(tfrs_to_grid_M(combined_pose).unsqueeze(0), size=[1, 2, h, w])
            
            nonrigid_model = DeeperHistRegModule(tensor_size=[h, w], disp_type="bspline", cp_spacing=(0.1, 0.1), splines_type="cubic", final_hw=[h, w])
            nonrigid_model.set_device(torch.device("cpu"))
            nonrigid_model.displacement_field = dict_nonrigid_model[f"{i_layer}.displacement_field"]
            disp_nonrigid = F.interpolate(nonrigid_model.disp.permute(0, 3, 1, 2).detach().cpu(), size=[h, w], mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            grid_nonrigid = disp_field_to_grid_2d(disp_nonrigid)
            l_grid_poses = [grid_combined, grid_nonrigid]

            combined_pose = torch.tensor([0, 0, 0, 0, 0,  1, 1]).float() if i_layer == 0 else combined_pose
            attr0.update({
                "tensor_tfrs_pose": combined_pose,
                "grid_M" : l_grid_poses
            })
            attr0.save_cache_layer()



    def plot_nchw_2d(
        self,
        l_layers = None,
        n_col = 10,
        n_row = 10,
        plt_figsize=(15, 15),
    ) -> plt.Figure:
        return plot_nchw_2d(
            om_data=self,
            l_layers=l_layers,
            n_col=n_col,
            n_row=n_row,
            plt_figsize=plt_figsize
        )


import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss is required for this function") from e


class GridFaissWarpCache:
    """
    Cache everything expensive per-grid:
    - Faiss index over source_coords (CPU or GPU)
    - target_coords grid (device)
    - grid as image for sampling (device)
    - grid gradients dg/du, dg/dv (device) for Gauss-Newton refinement
    """
    def __init__(self, grid_1hw2: torch.Tensor, use_gpu_faiss: bool = False):
        """
        grid_1hw2: [1, H, W, 2], normalized in [-1,1], same as used by grid_sample
        """
        assert grid_1hw2.ndim == 4 and grid_1hw2.shape[0] == 1 and grid_1hw2.shape[-1] == 2
        self.device = grid_1hw2.device
        self.H = int(grid_1hw2.shape[1])
        self.W = int(grid_1hw2.shape[2])

        # ---- source coords for Faiss (flattened), must be float32 on CPU for IndexFlatL2
        source_coords = grid_1hw2.view(-1, 2).detach()
        self.source_coords_cpu = source_coords.to("cpu", dtype=torch.float32).contiguous().numpy()

        # ---- build faiss index
        cpu_index = faiss.IndexFlatL2(2)
        cpu_index.add(self.source_coords_cpu)

        self.faiss_on_gpu = False
        self.index = cpu_index
        if use_gpu_faiss:
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    self.faiss_on_gpu = True
            except Exception:
                # Fallback to CPU index silently
                self.index = cpu_index
                self.faiss_on_gpu = False

        # ---- target coords (regular grid): affine_grid gives normalized [-1,1]
        # Note: align_corners=False to match your code
        target_coords = F.affine_grid(
            torch.eye(3, device=self.device, dtype=torch.float32)[:2, :].unsqueeze(0),
            (1, 1, self.H, self.W),
            align_corners=False
        ).view(-1, 2)  # [-1,1]
        self.target_coords = target_coords  # device float32

        # ---- grid as image for grid_sample: [1, 2, H, W]
        self.grid_img = grid_1hw2.permute(0, 3, 1, 2).contiguous().detach().to(dtype=torch.float32)

        # ---- precompute gradients in normalized coord space (u,v in [-1,1])
        # Compute central differences in pixel index space, then scale by dx/du, dy/dv.
        # For align_corners=False: x = ((u+1)*W - 1)/2  => dx/du = W/2
        #                       y = ((v+1)*H - 1)/2  => dy/dv = H/2
        with torch.no_grad():
            g = self.grid_img  # [1,2,H,W]

            # dg/dx in pixel-index units (x corresponds to W dimension)
            dg_dx = 0.5 * (g[..., 2:] - g[..., :-2])  # [1,2,H,W-2]
            dg_dx = F.pad(dg_dx, pad=(1, 1, 0, 0), mode="replicate")  # -> [1,2,H,W]

            # dg/dy in pixel-index units (y corresponds to H dimension)
            dg_dy = 0.5 * (g[:, :, 2:, :] - g[:, :, :-2, :])  # [1,2,H-2,W]
            dg_dy = F.pad(dg_dy, pad=(0, 0, 1, 1), mode="replicate")  # -> [1,2,H,W]

            # scale to dg/du, dg/dv where u,v are normalized [-1,1]
            self.dg_du_img = dg_dx * (self.W / 2.0)  # [1,2,H,W]
            self.dg_dv_img = dg_dy * (self.H / 2.0)  # [1,2,H,W]

    @torch.no_grad()
    def search_knn(self, query_uv_or_xy_m11: torch.Tensor, k: int):
        """
        query_uv_or_xy_m11: [N,2] float32, normalized [-1,1], on any device
        returns:
          indices_t: [N,k] long on self.device
          dists_t:   [N,k] float32 on self.device (squared L2 from faiss)
        """
        q = query_uv_or_xy_m11.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        dists, indices = self.index.search(q, k)
        indices_t = torch.from_numpy(indices).to(self.device, dtype=torch.long)
        dists_t = torch.from_numpy(dists).to(self.device, dtype=torch.float32)
        return indices_t, dists_t

    @torch.no_grad()
    def refine_inverse_map(
        self,
        init_uv_m11: torch.Tensor,     # [N,2] in [-1,1]
        target_source_xy_m11: torch.Tensor,  # [N,2] in [-1,1] (the "landmark" in source space)
        iters: int = 3,
        damping: float = 1e-4
    ) -> torch.Tensor:
        """
        Solve: grid(init_uv) ~= target_source_xy
        Using Gauss-Newton on u,v.
        """
        uv = init_uv_m11.clone()

        # grid_sample expects grid shape [N, Hout, Wout, 2]; we use [1, N, 1, 2]
        for _ in range(int(iters)):
            samp_grid = uv.view(1, -1, 1, 2)

            g_uv = F.grid_sample(
                self.grid_img, samp_grid,
                mode="bilinear", padding_mode="border", align_corners=False
            ).view(2, -1).t()  # [N,2]

            dg_du = F.grid_sample(
                self.dg_du_img, samp_grid,
                mode="bilinear", padding_mode="border", align_corners=False
            ).view(2, -1).t()  # [N,2]

            dg_dv = F.grid_sample(
                self.dg_dv_img, samp_grid,
                mode="bilinear", padding_mode="border", align_corners=False
            ).view(2, -1).t()  # [N,2]

            r = (g_uv - target_source_xy_m11)  # [N,2]

            # Jacobian per point:
            # [ dgx/du dgx/dv ]
            # [ dgy/du dgy/dv ]
            J11 = dg_du[:, 0]
            J21 = dg_du[:, 1]
            J12 = dg_dv[:, 0]
            J22 = dg_dv[:, 1]

            rx = r[:, 0]
            ry = r[:, 1]

            # A = J^T J + λI (2x2)
            A11 = J11 * J11 + J21 * J21 + damping
            A12 = J11 * J12 + J21 * J22
            A22 = J12 * J12 + J22 * J22 + damping

            # b = J^T r (2x1)
            b1 = J11 * rx + J21 * ry
            b2 = J12 * rx + J22 * ry

            det = A11 * A22 - A12 * A12
            det = torch.clamp(det, min=1e-12)

            # delta = A^{-1} b
            du = ( A22 * b1 - A12 * b2) / det
            dv = (-A12 * b1 + A11 * b2) / det

            uv[:, 0] = uv[:, 0] - du
            uv[:, 1] = uv[:, 1] - dv

            uv = torch.clamp(uv, -1.0, 1.0)

        return uv


def pad_kpt_to_align_HD(
    np_kpts: np.ndarray,
    gridM: torch.Tensor,
    max_size: int = 1280,
    k: int = 4,
    refine_iters: int = 3,
    use_gpu_faiss: bool = False,
    cache: GridFaissWarpCache | None = None
) -> np.ndarray:
    """
    High-definition inverse warp for keypoints without upsampling grid by interpolate.

    Args:
        np_kpts: [N,2] in pixel coords (0..max_size)
        gridM:   [B,H,W,2] or [1,H,W,2] grid used by grid_sample, normalized [-1,1]
        max_size: normalization denominator for keypoints (same as your original)
        k:        KNN for coarse init (>=2 recommended)
        refine_iters: Gauss-Newton refinement iterations (2~5 typical)
        use_gpu_faiss: enable GPU faiss if available
        cache: optionally pass a prebuilt cache to avoid rebuilding index/gradients

    Returns:
        np.ndarray [N,2] in pixel coords (0..max_size)
    """
    assert isinstance(np_kpts, np.ndarray) and np_kpts.ndim == 2 and np_kpts.shape[1] == 2

    if gridM.ndim == 4 and gridM.shape[0] != 1:
        grid_1hw2 = gridM[0:1]
    else:
        grid_1hw2 = gridM
    assert grid_1hw2.shape[0] == 1 and grid_1hw2.shape[-1] == 2

    device = grid_1hw2.device

    # Build / reuse cache (dominant speed win vs per-call rebuild)
    if cache is None:
        cache = GridFaissWarpCache(grid_1hw2, use_gpu_faiss=use_gpu_faiss)

    # 1) Normalize keypoints with fp64 math, then to [-1,1]
    kpts = torch.from_numpy(np_kpts).to(device=device, dtype=torch.float64)
    kpts01 = kpts / float(max_size)                         # [0,1] in float64
    kpts_m11 = (kpts01 * 2.0 - 1.0).to(torch.float32)       # [-1,1] float32 for faiss + torch ops

    # 2) Coarse init via Faiss on source_coords
    #    indices are into flattened grid positions.
    indices, dists = cache.search_knn(kpts_m11, k=max(k, 1))  # [N,k], [N,k]

    # coarse init in target uv: use weighted average of k neighbors in target space
    tgt_uv_knn = cache.target_coords[indices]  # [N,k,2] in [-1,1]
    if k <= 1:
        init_uv = tgt_uv_knn[:, 0, :]
    else:
        # weights: inverse distance (faiss returns squared L2)
        w = 1.0 / (dists + 1e-8)                         # [N,k]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        init_uv = (tgt_uv_knn * w.unsqueeze(-1)).sum(dim=1)  # [N,2]

    # 3) Refine by solving grid(uv) ~= source_xy
    if refine_iters and refine_iters > 0:
        uv_refined = cache.refine_inverse_map(
            init_uv_m11=init_uv,
            target_source_xy_m11=kpts_m11,
            iters=refine_iters,
            damping=1e-4
        )
    else:
        uv_refined = init_uv

    # 4) Back to [0,1] then pixel coords
    uv01 = 0.5 * (uv_refined + 1.0)  # [0,1]
    out = (uv01.to(torch.float64) * float(max_size)).cpu().numpy()
    return out
