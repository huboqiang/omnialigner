import os
from typing import Tuple, List
import torch
import numpy as np
import dask.array as da
from tqdm import tqdm

import omnialigner as om
from omnialigner.align.stack_func import calculate_flip_angle
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageTag, StageSampleTag
from omnialigner.keypoints.keypoints import kpts_flip, KeypointDetectorMeta
from omnialigner.utils.rtk_module import AffineEngineRawData
from omnialigner.utils.image_transform import apply_tfrs_to_dask
from omnialigner.utils import im2tensor, tensor2im
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw, DataType

def stack(om_data: om.Omni3D,
          l_layers: List[int]=None,
          detector: KeypointDetectorMeta=None,
          is_stack_via_keypoints: bool=False,
          overwrite_cache:bool=False
    ) -> Tuple[Tensor_image_NCHW, List[Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]]]:
    """Stack and pre-align a sequence of images for further alignment processing.

    This function performs initial alignment of image sequences by handling flipping,
    rotation, and keypoint matching between consecutive frames. It supports both
    keypoint-based and direct stacking approaches.

    The stacking process involves:
    1. Loading and validating padded images
    2. Detecting and matching keypoints between frames
    3. Calculating optimal flip and rotation parameters
    4. Applying transformations to align consecutive frames
    5. Caching results for future use

    Args:
        om_data (om.Omni3D): Omni3D object containing:
            - Image data and project configuration
            - Padding information
            - Cache file paths and settings
        l_layers (List[int], optional): List of layer indices to process.
            If None, processes all layers. Defaults to None.
        detector (KeypointDetectorMeta, optional): Keypoint detector instance.
            If None, uses default detector. Defaults to None.
        is_stack_via_keypoints (bool, optional): Whether to use keypoint-based
            stacking for refined alignment. Recommended for small sequences.
            Defaults to False.
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.

    Returns:
        tuple:
            - stacked_tensor (Tensor_image_NCHW): Stacked and pre-aligned images
              with shape [N, C, H, W] where:
                - N: number of layers
                - C: number of channels
                - H, W: height and width after padding
            - l_kpts_pairs (List[Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]]):
              List of keypoint pairs between consecutive frames, where each pair
              contains coordinates for matching points.

    Raises:
        RuntimeError: If padded tensor is not found (requires prior om.pad() execution).

    Note:
        - Requires prior execution of om.pad()
        - Results are cached by default for efficiency
        - Keypoint-based stacking creates additional cache in '04.detect_kpts/omni_stack'
        - Supports both regular and keypoint-based stacking methods
        - Handles special cases for HE (H&E stained) images
    """
    logging.info(f"Processing stack, stage1, check inputs")
    FILE_cache_pad = StageTag["PAD"]
    dict_file_name = FILE_cache_pad.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name is None:
        logging.error("Padded tensor not found. Please run `om.pad()` first.")
        return None, None
    
    
    logging.info(f"Processing stack, stage2, return cache if exists")
    FILE_cache_stack = StageTag["STACK"]
    dict_file_name = FILE_cache_stack.get_file_name(projInfo=om_data.proj_info)
    if dict_file_name is not None and not overwrite_cache:
        stacked_tensor = torch.load(dict_file_name["padded_tensor"])
        l_kpts_pairs = torch.load(dict_file_name["l_kpts_pairs"])
        if l_layers is not None:
            stacked_tensor = stacked_tensor[l_layers, :, :, :]
            l_kpts_pairs = [l_kpts_pairs[i_layer] for i_layer in l_layers]
            return stacked_tensor, l_kpts_pairs

        return stacked_tensor, l_kpts_pairs

    logging.info(f"Processing stack, stage3, expand l_layers")
    dict_file_name = FILE_cache_stack.get_file_name(projInfo=om_data.proj_info, check_exist=False)
    
    if l_layers is None:
        N_layers = len(om_data)
        l_layers = range(N_layers)
    else:
        overwrite_cache = False
    
    logging.info(f"Processing stack, stage4, running stack")
    padded_tensors = om_data.load_3d_NCHW("PAD")
    detector = om.kp.init_detector(detector)
    
    if is_stack_via_keypoints:
        return stack_via_keypoints(om_data=om_data, padded_tensor=padded_tensors, detector=detector, overwrite_cache=overwrite_cache)

    l_res = calculate_flip_angle(om_data, l_layers=l_layers, detector=detector, overwrite_cache=False)
    FILE_cache_kpts = StageSampleTag["KEYPOINTS"]
    l_images = []
    l_kpts_pairs = []
    N_layers = len(l_layers)
    for i_layer in tqdm(range(N_layers-1), desc="Processing frames"):
        idx_curr, idx_next = l_layers[i_layer], l_layers[i_layer+1]
        angle, flip, _ = l_res[i_layer+1]
        
        kpts_curr, kpts_next_left = load_keypoint_pairs(om_data, idx_curr, FILE_cache_kpts)
        kpts_next_right = torch.FloatTensor([])
        if i_layer < N_layers-2:
            kpts_next_right, _ = load_keypoint_pairs(om_data, idx_next, FILE_cache_kpts)

        tensor_curr = padded_tensors[idx_curr:idx_curr+1]
        tensor_next = padded_tensors[idx_next:idx_next+1]
        
        if i_layer == 0:
            l_images.append(tensor_curr)
            l_kpts_pairs.append([torch.FloatTensor([]), kpts_curr])
        
        kwargs_pad = {"mode": "constant"}
        if om_data.proj_info.get_dtype(i_layer) == "HE" and om_data.tag == DataType.RAW:
            kwargs_pad["constant_values"] = 255
        
        tensor_next_moved, kpts_next_left_moved, kpts_next_right_moved = process_image_and_keypoints(
            tensor_next, kpts_next_left, kpts_next_right, flip, angle, om_data.max_size, **kwargs_pad
        )
        
        l_images.append(tensor_next_moved)
        l_kpts_pairs.append([
            kpts_next_left_moved,
            kpts_next_right_moved
        ])

    logging.info(f"Processing stack, stage5, save cache {overwrite_cache}")
    stacked_tensor = torch.concat(l_images)
    l_kpts_pairs_interval = [ [l_kpts_pairs[idx][1], l_kpts_pairs[idx+1][0]] for idx in range(len(l_kpts_pairs)-1) ]
    if overwrite_cache:
        logging.info(f"write to {dict_file_name}")
        torch.save(stacked_tensor, dict_file_name["padded_tensor"], _use_new_zipfile_serialization=False)
        torch.save(l_res, dict_file_name["flip_angle"], _use_new_zipfile_serialization=False)
        torch.save(l_kpts_pairs_interval, dict_file_name["l_kpts_pairs"], _use_new_zipfile_serialization=False)
    
    return stacked_tensor, l_kpts_pairs_interval


def load_keypoint_pairs(
        omni: om.Omni3D,
        i_layer: int, 
        FILE_cache_kpts: StageSampleTag) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]:
    """Load matched keypoint pairs from cache for a specific layer.

    This function retrieves pre-computed keypoint pairs from the cache files.
    The keypoints represent corresponding points between consecutive frames
    that have been matched during keypoint detection.

    Args:
        omni (om.Omni3D): Omni3D object containing project configuration.
        i_layer (int): Layer index to load keypoints from.
        FILE_cache_kpts (StageSampleTag): Cache tag for keypoint files.

    Returns:
        tuple:
            - kpts_F (Tensor_kpts_N_xy_raw): Fixed frame keypoints [N, 2].
            - kpts_M (Tensor_kpts_N_xy_raw): Moving frame keypoints [N, 2].

    Note:
        - The returned keypoints are already matched pairs, filtered by the
          matching process during keypoint detection
        - Keypoints are loaded from cache files in the format specified by
          the project configuration
        - The function expects the cache files to exist and contain valid data
    """
    file_kpts = FILE_cache_kpts.get_file_name(
        projInfo=omni.proj_info, i_layer=i_layer
    )
    pt_res = torch.load(file_kpts["kpts"])
    kpts_M = pt_res["test_input"][pt_res["index_matches"]]
    kpts_F = pt_res["test_label"][pt_res["index_matches"]]
    
    return kpts_F, kpts_M

def process_image_and_keypoints(
        image: Tensor_image_NCHW,
        kpts_left: Tensor_kpts_N_xy_raw,
        kpts_right: Tensor_kpts_N_xy_raw,
        flip: Tuple[bool, bool],
        angle: float,
        max_size: int,
        **kwargs_pad
    ) -> Tuple[Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]:
    """Transform an image and its keypoints using flipping and rotation.

    This function applies geometric transformations (flipping and rotation) to both
    the image and its associated keypoints to maintain consistency between image
    and keypoint transformations.

    Args:
        image (Tensor_image_NCHW): Input image tensor [1, C, H, W].
        kpts_left (Tensor_kpts_N_xy_raw): Left keypoints [N, 2].
        kpts_right (Tensor_kpts_N_xy_raw): Right keypoints [N, 2].
        flip (Tuple[bool, bool]): (flip_x, flip_y) flags for flipping.
        angle (float): Rotation angle in degrees.
        max_size (int): Maximum size constraint for output dimensions.
        **kwargs_pad: Additional padding parameters, including:
            - mode (str): Padding mode ('constant', 'reflect', etc.)
            - constant_values (int): Fill value for 'constant' mode

    Returns:
        tuple:
            - image_rotated (Tensor_image_NCHW): Transformed image [1, C, H, W].
            - kpts_left_rotated (Tensor_kpts_N_xy_raw): Transformed left keypoints [N, 2].
            - kpts_right_rotated (Tensor_kpts_N_xy_raw): Transformed right keypoints [N, 2].

    Note:
        - Flipping is applied first, followed by rotation
        - Transformations maintain consistency between image and keypoint coordinates
        - Uses dask arrays for memory-efficient processing of large images
        - Special handling for padding modes and values based on image type
        - Tile size for dask operations is optimized for memory usage (2048x2048)
    """
    kpts_left_flipped = kpts_flip(kpts_left, image=image, flip_x=flip[0], flip_y=flip[1])
    kpts_right_flipped = kpts_flip(kpts_right, image=image, flip_x=flip[0], flip_y=flip[1])
    # image_flipped = image_flip(image, flip_x=flip[0], flip_y=flip[1])    
    # if angle == 0:
    #     return image_flipped, kpts_left_flipped, kpts_right_flipped

    # stn = AffineEngineRawData(
    #     init_tfrs=torch.FloatTensor([angle/180*np.pi, 0., 0., 0., 0., 1, 1]),
    #     params_hw=[max_size, max_size]
    # )
    # with torch.no_grad():
    #     image_rotated = stn(image_flipped)

    da_img = da.from_array(tensor2im(image))
    flip_x = 1 if flip[0] == 0 else -1
    flip_y = 1 if flip[1] == 0 else -1
    tensor_tfrs=[[angle/180*np.pi, 0., 0., 0., 0., flip_x, flip_y]]
    da_image_rotated = apply_tfrs_to_dask(da_image=da_img, tensor_tfrs=tensor_tfrs, tile_size=(2048, 2048, 1), **kwargs_pad)
    image_rotated = im2tensor(da_image_rotated)
    stn = AffineEngineRawData(
        init_tfrs=torch.FloatTensor([-angle/180*np.pi, 0., 0., 0., 0., 1, 1]),
        params_hw=[max_size, max_size]
    )
    with torch.no_grad():
        kpts_left_rotated = stn.stn_keypoints(kpts_left_flipped.clone())
        kpts_right_rotated = stn.stn_keypoints(kpts_right_flipped.clone())
    return image_rotated, kpts_left_rotated, kpts_right_rotated


def load_file(file_name, max_size=1600):
    """Load a tensor file with size constraints.

    This function loads a tensor from a file and applies size constraints to ensure
    the output dimensions do not exceed the specified maximum size. This is useful
    for memory management when working with large images.

    Args:
        file_name (str): Path to the tensor file.
        max_size (int, optional): Maximum size constraint for both dimensions.
            Defaults to 1600.

    Returns:
        torch.Tensor: Loaded and size-constrained tensor with shape [N, C, H', W']
            where H' = min(H, max_size) and W' = min(W, max_size).

    Note:
        - The function preserves the batch and channel dimensions
        - Only height and width dimensions are constrained
        - Original tensor shape is preserved if smaller than max_size
        - Useful for preventing memory issues with large images
    """
    tensor = torch.load(file_name)
    _, _, H, W = tensor.shape

    H = min(H, max_size)
    W = min(W, max_size)
    return tensor[:, :, 0:H, 0:W]


def stack_via_keypoints(
        om_data: om.Omni3D,
        padded_tensor: Tensor_image_NCHW,
        detector: KeypointDetectorMeta=None,
        overwrite_cache: bool=False
    ) -> Tuple[Tensor_image_NCHW, List[Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]]]:
    """Stack images using keypoint-based alignment for improved accuracy.

    This function performs a more refined stacking process by using keypoint
    detection and matching to calculate optimal transformation parameters
    between consecutive frames. This approach is particularly useful for
    small sequences where precise alignment is critical.

    The process involves:
    1. Detecting keypoints in each frame
    2. Matching keypoints between consecutive frames
    3. Computing optimal transformation parameters (TFRS)
    4. Applying transformations to align frames
    5. Caching results for future use

    Args:
        om_data (om.Omni3D): Omni3D object containing project data and settings.
        padded_tensor (Tensor_image_NCHW): Pre-padded image tensor [N, C, H, W].
        detector (KeypointDetectorMeta, optional): Keypoint detector instance.
            If None, uses default detector. Defaults to None.
        overwrite_cache (bool, optional): Whether to force recomputation.
            Defaults to False.

    Returns:
        tuple:
            - stacked_tensor (Tensor_image_NCHW): Stacked and aligned images
              with shape [N, C, H, W].
            - l_kpts_pairs (List[Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]]):
              List of keypoint pairs between consecutive frames.

    Note:
        - Creates additional cache in '04.detect_kpts/omni_stack'
        - Recommended for small sequences (5-10 images)
        - More computationally intensive than basic stacking
        - Provides better initialization for affine/nonrigid training
        - Handles image channel expansion for compatibility with detectors
        - Caches intermediate results for efficiency in reprocessing
        - Supports transformation parameter caching for later use
    """
    stacked_tensor = padded_tensor.clone()
    detector = om.kp.init_detector(detector)
    
    l_stacked_tfrs = [ torch.FloatTensor([0,0,0,0,0, 1,1]).float() ]
    l_stacked_kpt_pairs = []
    for idx in range(len(om_data)-1):
        img_F = torch.repeat_interleave(stacked_tensor[idx+0:idx+1, :, :, :], 3, 1)
        img_M = torch.repeat_interleave(stacked_tensor[idx+1:idx+2, :, :, :], 3, 1)
        sample_F = om_data.proj_info.get_sample_name(idx+0)
        sample_M = om_data.proj_info.get_sample_name(idx+1)

        dict_file_name_stack = StageTag.STACK.get_file_name(projInfo=om_data.proj_info, check_exist=False)
        dir_stack = os.path.dirname(dict_file_name_stack["padded_tensor"])
        dir_stack_tfrs = f"{dir_stack}/omni_stack/"
        file_kpt = f"{dir_stack_tfrs}/{sample_F}_{sample_M}.pth"
        os.makedirs(dir_stack_tfrs, exist_ok=True)

        if os.path.isfile(file_kpt) and not overwrite_cache:
            kd = om.kp.KeypointPairs()
            kd.dataset = torch.load(file_kpt)
        else:
            kd = om.kp.detect_AngleFlipScale(image_F=img_F, image_M=img_M, detector=detector)
            torch.save(kd.dataset, file_kpt)

        img_M_moved = kd.move_img_M()
        tensor_tfrs = kd.calculate_tfrs()

        kpt_F = kd.dataset["train_label"]
        kpt_M = kd.move_kpt_M()
        stacked_tensor[idx+1:idx+2] = om.tl.im2tensor(img_M_moved)[:, 0:1, :, :]

        l_stacked_tfrs.append(tensor_tfrs)
        l_stacked_kpt_pairs.append([kpt_F, kpt_M])

    logging.info(f"Processing stack, stage5, save cache {overwrite_cache}")
    if overwrite_cache:
        logging.info(f"write to {dict_file_name_stack}")
        torch.save(stacked_tensor, dict_file_name_stack["padded_tensor"], _use_new_zipfile_serialization=False)
        torch.save(l_stacked_kpt_pairs, dict_file_name_stack["l_kpts_pairs"], _use_new_zipfile_serialization=False)
        torch.save(l_stacked_tfrs, f"{dir_stack_tfrs}/tfrs_to_stacked.pt", _use_new_zipfile_serialization=False)
        torch.save([], f"{dir_stack_tfrs}/../flip_angle.pt", _use_new_zipfile_serialization=False)

    return stacked_tensor, l_stacked_kpt_pairs
    