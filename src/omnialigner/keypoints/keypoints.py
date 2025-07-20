from typing import List, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import faiss

from omnialigner.keypoints.keypoint_pairs import KeypointPairs
from omnialigner.dtypes import KeypointDetectorMeta
from omnialigner.dtypes import Tensor_image_NCHW, Np_image_HWC, Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed, Np_kpts_N_yx_raw, Dask_image_HWC
from omnialigner.utils.affine_module import AffineEngineRawData

def detect(
        image: Tensor_image_NCHW, 
        detector: KeypointDetectorMeta=None
    ) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed]:
    """Detect keypoints and their embeddings in an image.

    Args:
        image: Input image tensor in NCHW format.
        detector: Optional keypoint detector instance. If None, uses default detector.

    Returns:
        tuple:
            - Tensor_kpts_N_xy_raw: Detected keypoint coordinates in (x, y) format.
            - Tensor_kpts_N_embed: Feature embeddings for each keypoint.
    """
    detector = init_default_detector(detector)
    return detector.detect(image)

def match(
        image_F: Dask_image_HWC|Tensor_image_NCHW, 
        image_M: Dask_image_HWC|Tensor_image_NCHW, 
        detector: KeypointDetectorMeta=None, 
        **kwargs
    ) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw, List[int], Np_image_HWC]:
    """Match keypoints between two images using the specified detector.

    Args:
        image_F: Fixed (reference) image in either Dask or Tensor format.
        image_M: Moving image to be aligned in either Dask or Tensor format.
        detector: Optional keypoint detector instance. If None, uses default detector.
        **kwargs: Additional arguments passed to the detector's match method.

    Returns:
        tuple:
            - tensor_kpts_F: Keypoints detected in the fixed image.
            - tensor_kpts_M: Matched keypoints in the moving image.
            - l_idxs: List of matching indices between keypoint pairs.
            - MatchedCanvas: Visualization of the matched keypoints.
    """
    detector = init_default_detector(detector)
    return detector.match(image_F, image_M, **kwargs)


def select_angles(
        image_F: Tensor_image_NCHW,
        image_M: Tensor_image_NCHW, 
        n_angles: int=1, 
        detector: KeypointDetectorMeta=None
    ) -> KeypointPairs:
    """Find optimal rotation angle for image alignment based on keypoint matching.

    Tests different rotation angles and selects the one that yields the most matched keypoints.

    Args:
        image_F: Fixed (reference) image tensor.
        image_M: Moving image tensor to be rotated.
        n_angles: Number of angles to test. Will test angles in range(0, 360, 360//n_angles).
        detector: Optional keypoint detector instance. If None, uses default detector.

    Returns:
        KeypointPairs: Object containing the best matching results, including:
            - Matched keypoints in both images
            - Best rotation angle
            - Transformed moving image
    """
    detector = init_default_detector(detector)
    
    best_mkpts_F = torch.FloatTensor([])
    best_mkpts_M = torch.FloatTensor([])
    best_matches = 0
    l_idxs = []
    kd = KeypointPairs(image_F=image_F, image_M=image_M, mkpts_F=best_mkpts_F, mkpts_M=best_mkpts_M, index_matches=l_idxs)
    for angle in range(0, 360, 360//n_angles):
        stn = AffineEngineRawData(init_tfrs=torch.FloatTensor([angle/180*np.pi, 0., 0., 0.,0., 1., 1.]), params_hw=[image_M.shape[2], image_M.shape[3]])
        tensor = stn(image_M)
        image_move = tensor.detach()
        # try:
        mkpts_F, mkpts_M, matches, canvas = detector.forward(image_F, image_move)
        mkpts_F = mkpts_F.cpu()
        mkpts_M = mkpts_M.cpu()
        if len(matches) > best_matches:
            l_idxs = [ m for m in matches ]
            best_mkpts_F = mkpts_F
            stn_kpt = AffineEngineRawData(init_tfrs=torch.FloatTensor([angle/180*np.pi, 0., 0., 0.,0., 1., 1.]), params_hw=[image_M.shape[2], image_M.shape[3]])
            mkpts_M_ = stn_kpt.stn_keypoints(mkpts_M.clone())
            best_mkpts_M = mkpts_M_
            best_matches = len(matches)

            kd.update_kpts(best_mkpts_F, best_mkpts_M, l_idxs)
            kd.dataset["best_angle"] = angle
            kd.image_M_adjusted = image_move

    return kd

def select_angles_with_scales(
        image_F: Tensor_image_NCHW,
        image_M: Tensor_image_NCHW,
        n_angles: int=8,
        detector: KeypointDetectorMeta=None,
        scale_ratios=[1, 2, 4, 8]
    ) -> KeypointPairs:
    """Find optimal rotation angle and scale for image alignment.

    Tests combinations of rotation angles and scales to find the best alignment parameters
    based on keypoint matching.

    Args:
        image_F: Fixed (reference) image tensor.
        image_M: Moving image tensor to be transformed.
        n_angles: Number of angles to test. Will test angles in range(0, 360, 360//n_angles).
        detector: Optional keypoint detector instance. If None, uses default detector.
        scale_ratios: List of scale factors to test for image resizing.

    Returns:
        KeypointPairs: Object containing the best matching results, including:
            - Matched keypoints in both images
            - Best rotation angle and scale
            - Transformed moving image
    """
    detector = init_default_detector(detector)
    best_mkpts_F = torch.FloatTensor([])
    best_mkpts_M = torch.FloatTensor([])
    best_idxs = []

    kd = KeypointPairs(image_F=image_F, image_M=image_M, mkpts_F=best_mkpts_F, mkpts_M=best_mkpts_M, index_matches=best_idxs)

    for sr in scale_ratios:
        img_size = (image_M.shape[2], image_M.shape[3])
        img_size = (img_size[0] // sr, img_size[1] // sr)
        image_M_ = F.interpolate(image_M, img_size, mode='bilinear', align_corners=True)
        image_F_ = F.interpolate(image_F, img_size, mode='bilinear', align_corners=True)
        
        scale_ratio_F = np.array([image_M.shape[3] / image_F.shape[3], image_M.shape[2] / image_F.shape[2]])
        kd_angle = select_angles(image_F_, image_M_, n_angles, detector=detector)
        if len(kd_angle.dataset["index_matches"]) > len(best_idxs):
            best_mkpts_F = kd_angle.dataset["test_label"] * sr / scale_ratio_F
            best_mkpts_M = kd_angle.dataset["test_input"] * sr
            best_idxs = kd_angle.dataset["index_matches"]
            best_angle = kd_angle.dataset["best_angle"]
            kd.update_kpts(best_mkpts_F, best_mkpts_M, best_idxs)
            kd.dataset["best_angle"] = best_angle
            kd.image_M_adjusted = kd_angle.image_M_adjusted

    return kd


def select_angles_with_scales_flip(
        image_F: Tensor_image_NCHW,
        image_M: Tensor_image_NCHW,
        n_angles: int=8,
        detector: KeypointDetectorMeta=None,
        scale_ratios=[1, 2, 4, 8]
    ) -> KeypointPairs:
    """Find optimal rotation, scale, and flip parameters for image alignment.

    Tests combinations of rotation angles, scales, and flip operations to find
    the best alignment parameters based on keypoint matching.

    Args:
        image_F: Fixed (reference) image tensor.
        image_M: Moving image tensor to be transformed.
        n_angles: Number of angles to test. Will test angles in range(0, 360, 360//n_angles).
        detector: Optional keypoint detector instance. If None, uses default detector.
        scale_ratios: List of scale factors to test for image resizing.

    Returns:
        KeypointPairs: Object containing the best matching results, including:
            - Matched keypoints in both images
            - Best rotation angle, scale, and flip parameters
            - Transformed moving image
    """
    detector = init_default_detector(detector)
    best_mkpts_F = torch.FloatTensor([])
    best_mkpts_M = torch.FloatTensor([])
    best_idxs = []

    kd = KeypointPairs(image_F=image_F, image_M=image_M, mkpts_F=best_mkpts_F, mkpts_M=best_mkpts_M, index_matches=best_idxs)
    
    for flip_x in [0, 1]:
        for flip_y in [0, 1]:
            image_M_ = image_flip(image_M, flip_y=flip_y, flip_x=flip_x)
            kd_flip = select_angles_with_scales(image_F, image_M_, n_angles, detector=detector, scale_ratios=scale_ratios)
            if len(kd_flip.dataset["index_matches"]) > len(best_idxs):
                best_flip = [flip_x, flip_y]
                best_image = kd_flip.image_M_adjusted
                best_mkpts_F = kd_flip.dataset["test_label"]
                best_mkpts_M = kd_flip.dataset["test_input"]
                best_idxs = kd_flip.dataset["index_matches"]
                best_angle = kd_flip.dataset["best_angle"]
                best_mkpts_M_ = kpts_flip(best_mkpts_M, image=image_M, flip_x=flip_x, flip_y=flip_y)
                kd.update_kpts(best_mkpts_F, best_mkpts_M_, best_idxs)
                kd.dataset["best_angle"] = best_angle
                kd.image_M_adjusted = best_image
                kd.dataset["flip"] = best_flip
                
    
    # best_mkpts_M = kpts_flip(best_mkpts_M, image=image_M, flip_x=best_flip[0], flip_y=best_flip[1])
    return kd


def image_flip(image: Tensor_image_NCHW, flip_y: bool, flip_x: bool) -> Tensor_image_NCHW:
    """Flip an image horizontally and/or vertically.

    Args:
        image: Input image tensor in NCHW format.
        flip_y: Whether to flip the image vertically.
        flip_x: Whether to flip the image horizontally.

    Returns:
        Tensor_image_NCHW: Flipped image tensor.
    """
    axis = [i+2 for i, flip in enumerate([flip_y, flip_x]) if flip]
    if axis:
        return torch.flip(image, axis)

    return image

def kpts_flip(mkpts: Tensor_kpts_N_xy_raw, image: Tensor_image_NCHW, flip_y: bool, flip_x: bool) -> Tensor_kpts_N_xy_raw:
    """Flip keypoint coordinates according to image flip parameters.

    Args:
        mkpts: Keypoint coordinates in (x, y) format.
        image: Reference image tensor for dimension information.
        flip_y: Whether to flip coordinates vertically.
        flip_x: Whether to flip coordinates horizontally.

    Returns:
        Tensor_kpts_N_xy_raw: Flipped keypoint coordinates.
    """
    H, W = image.shape[2], image.shape[3]
    if mkpts.shape[0] == 0 or (flip_x + flip_y == 0):
        return mkpts
    
    mkpts = mkpts.clone()
    if flip_y:
        mkpts[:, 1] = H - mkpts[:, 1]
    
    if flip_x:
        mkpts[:, 0] = W - mkpts[:, 0]
    
    
    return mkpts

def init_default_detector(detector: str|KeypointDetectorMeta=None) -> KeypointDetectorMeta:
    """Initialize a keypoint detector instance.

    If no detector is provided, initializes a default detector (XfeatDetector).
    If a string is provided, initializes the corresponding detector type.

    Args:
        detector: Either a detector instance, a string specifying detector type,
            or None to use default.

    Returns:
        KeypointDetectorMeta: Initialized keypoint detector instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if detector is None:
        from omnialigner.keypoints.xfeat import XfeatDetector
        return XfeatDetector(device=device)
    
    
    if isinstance(detector, str):
        if detector == "xfeat":
            from omnialigner.keypoints.xfeat import XfeatDetector
            return XfeatDetector(device=device)
        if detector.lower()[0:4] == "roma":
            from omnialigner.keypoints.roma_dense import RomaDenseDetector
            return RomaDenseDetector(device=device)
        
    if isinstance(detector, KeypointDetectorMeta):
        return detector

    raise ValueError(f"Detector {detector} not supported")


def search_kpts(kpts_dense_M: Np_kpts_N_yx_raw|Tensor_kpts_N_xy_raw, 
                kpts_dense_F: Np_kpts_N_yx_raw|Tensor_kpts_N_xy_raw, 
                kpts_query_M: Np_kpts_N_yx_raw|Tensor_kpts_N_xy_raw, 
                k: int=5):
    """Search for nearest neighbor keypoints using FAISS.

    Performs k-nearest neighbor search between keypoint sets using FAISS indexing.

    Args:
        kpts_dense_M: Dense keypoints from moving image.
        kpts_dense_F: Dense keypoints from fixed image.
        kpts_query_M: Query keypoints from moving image to match.
        k: Number of nearest neighbors to find. Defaults to 5.

    Returns:
        tuple:
            - distances: Array of distances to k-nearest neighbors.
            - indices: Array of indices of k-nearest neighbors.
    """
    if isinstance(kpts_dense_M, torch.Tensor):
        kpts_dense_M = kpts_dense_M.cpu().numpy()

    if isinstance(kpts_dense_F, torch.Tensor):
        kpts_dense_F = kpts_dense_F.cpu().numpy()
        
    if isinstance(kpts_query_M, torch.Tensor):
        kpts_query_M = kpts_query_M.cpu().numpy()
        
    kpts_M = np.ascontiguousarray(kpts_dense_M.astype(np.float32))
    kpts_F = np.ascontiguousarray(kpts_dense_F.astype(np.float32))
    kpts_query_M = np.ascontiguousarray(kpts_query_M.astype(np.float32))
    
    # Create a FAISS index for L2 distance
    index = faiss.IndexFlatL2(2)  # Use '2' since each keypoint has 2 dimensions
    index.add(kpts_M)

    # Find the 5 nearest neighbors
    D, I = index.search(kpts_query_M, k)

    # Initialize an array to store the average coordinates
    average_coords_M = np.zeros((kpts_query_M.shape[0], 2))
    
    # Calculate the average coordinates for each query point in kpts_M
    for i, indices in enumerate(I):
        average_coords_M[i] = kpts_M[indices].mean(axis=0)

    # If you have a way to correlate indices to kpts_F, compute their averages here:
    # This is an assumption; you would need to adjust this according to your data's relationship:
    average_coords_F = np.zeros((kpts_query_M.shape[0], 2))
    if kpts_M.shape == kpts_F.shape:  # Example condition, replace with actual logic
        for i, indices in enumerate(I):
            average_coords_F[i] = kpts_F[indices].mean(axis=0)

    return average_coords_M, average_coords_F