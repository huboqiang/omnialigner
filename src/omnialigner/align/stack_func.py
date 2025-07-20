from typing import List, Tuple
from tqdm import tqdm
import torch
import numpy as np

import omnialigner as om
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageSampleTag
from omnialigner.keypoints.keypoints import kpts_flip
from omnialigner.utils.quaternion import combine_image_transformations, quaternion_multiply, quaternion_to_angle_flip

def calculate_flip_angle(
        om_data: om.Omni3D, 
        l_layers: List[int]=None, 
        detector: torch.nn.Module=None,
        overwrite_cache: bool=False
    ) -> List[Tuple[float, Tuple[int, int], float]]:
    detector = om.kp.init_detector(detector)
    l_angles = []
    l_flips = []
    flip = [0, 0]
    angle = 0
    if l_layers is None:
        l_layers = range(len(om_data))

    n_angle = om_data.config['kpt'].get("n_angles", 20)
    
    padded_tensor = om_data.padded_tensor
    N_layers = len(l_layers)
    kpt_tag = "KEYPOINTS"
    q_combined = combine_image_transformations(angle=0, flip=[0, 0])
    for i_layer in tqdm(range(N_layers-1)):
        try:
            FILE_per_sample = StageSampleTag[kpt_tag]
        except KeyError:
            raise ValueError(f"Invalid kpt_tag: {kpt_tag}. Must be one of: {[tag.name for tag in StageSampleTag]}")
        
        layer_curr = l_layers[i_layer]
        layer_next = l_layers[i_layer+1]
            
        file_kpts = FILE_per_sample.get_file_name(projInfo=om_data.proj_info, i_layer=layer_curr)
        image_tar = padded_tensor[layer_curr:layer_curr+1]
        if file_kpts is None or overwrite_cache:
            image_src = padded_tensor[layer_next:layer_next+1]
            kd = calculate_kpts(image_tar, image_src, detector=detector, n_angle=n_angle)
            file_kpts = FILE_per_sample.get_file_name(projInfo=om_data.proj_info, i_layer=layer_curr, check_exist=False)
            logging.info(f"keypoints file not found. calculated: {file_kpts}")
            fig = kd.plot_dataset()
            fig.savefig(f"{file_kpts['kpts']}.png")

            del kd.dataset["image_input"]
            del kd.dataset["image_label"]
            torch.save(kd.dataset, file_kpts["kpts"], _use_new_zipfile_serialization=False)

        pt_res = torch.load(file_kpts["kpts"])
        flip_ = pt_res["flip"]
        idx_0 = pt_res["index_matches"]
        kpt_F = pt_res["test_label"][idx_0, :].float()
        kpt_M = pt_res["test_input"][idx_0, :].float()
        kpt_M_flip = kpts_flip(kpt_M, image=image_tar, flip_x=flip_[0], flip_y=flip_[1])
        angle_ = calculate_angle(kpt_F, kpt_M_flip)
        q_combined_ = combine_image_transformations(angle=angle_, flip=flip_)
        q_combined = quaternion_multiply(q_combined, q_combined_)
        angle, flip = quaternion_to_angle_flip(q_combined)
        
        l_angles.append(angle)
        l_flips.append(flip)
    
    l_res = [ [0, [0, 0], 0]]
    i_layer = 0
    
    smooth_func = smooth_angles_short_range if len(l_angles) < 3 else smooth_angles
    use_smooth = om_data.config['kpt'].get("smooth_angle", True)
    if not use_smooth:
        smooth_func = lambda x: x

    np_angles_smooth = smooth_func(np.array(l_angles))
    for angle, flip, angle_raw in zip(np_angles_smooth, l_flips, l_angles):
        l_res.append([angle, flip, angle_raw])
        i_layer += 1
    
    return l_res


def smooth_angles_short_range(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Use sliding window median to smooth angle sequence
    
    Args:
        angles: original angle sequence
        window: sliding window size
    
    Returns:
        smoothed angle sequence
    """
    half_window = window // 2
    angles_smooth = np.zeros_like(angles)

    for i in range(len(angles)):
        start = max(0, i - half_window)
        end = min(len(angles), i + half_window + 1)        
        angles_smooth[i] = np.median(angles[start:end])
    
    return angles_smooth


def smooth_angles(angles: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Use sliding window median to smooth angle sequence
    
    Args:
        angles: original angle sequence
        window: sliding window size
    
    Returns:
        smoothed angle sequence
    """
    half_window = window // 2
    angles_smooth = np.zeros_like(angles)
    
    for i in range(len(angles)):
        start = max(0, i - half_window)
        end = min(len(angles), i + half_window + 1)        
        angles_smooth[i] = angles[i] - np.median(angles[start:end])
    
    return angles_smooth


def calculate_angle(kpt_F, kpt_M):
    A = np.hstack((kpt_M, np.ones((kpt_M.shape[0], 1))))
    solution, _, _, _ = np.linalg.lstsq(A, kpt_F, rcond=None)
    angle_rad = np.arctan2(solution[1, 0], solution[0, 0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_kpts(image_tar, image_src, detector=None, n_angle=20):
    kd_raw = om.kp.detect_AngleFlipScale(image_tar, image_src, n_angles=n_angle, detector=detector)
    flip = kd_raw.dataset["flip"]
    best_angle = kd_raw.dataset["best_angle"]
    if flip == [1, 1] and (best_angle > 90 and best_angle < 270):
        flip = [0, 0]
        best_angle = (best_angle - 180) % 360

    kd_raw.image_M_adjusted = None

    kd_raw.dataset["flip"] = flip
    kd_raw.dataset["best_angle"] = best_angle
    return kd_raw

