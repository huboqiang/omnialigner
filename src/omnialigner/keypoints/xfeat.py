import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from omnialigner.dtypes import KeypointDetectorMeta
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed, Np_image_HWC
from omnialigner.utils.image_viz import tensor2im
root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
vendor_path = os.path.join(root_path, 'vendor/accelerated_features/')
sys.path.append(vendor_path)

from omnialigner.vendor.accelerated_features.modules.xfeat import XFeat
torch.use_deterministic_algorithms(False)

class XfeatDetector(nn.Module, KeypointDetectorMeta):
    def __init__(self, **kwargs):
        super().__init__()
        if "device" in kwargs:
            del kwargs["device"]
        self.xfeat = XFeat(**kwargs)

    def detect(self, image: Tensor_image_NCHW) -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_embed]:
        out1 = self.xfeat.detectAndComputeDense(image)
        return out1['keypoints'], out1['descriptors']

    def match(self, image_F: Tensor_image_NCHW, image_M: Tensor_image_NCHW, method: str="xfeats_cv2DMatch") -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw, List[int], Np_image_HWC]:
        if method == "xfeats_refineMatch":
            out1 = self.xfeat.detectAndComputeDense(image_F)
            out2 = self.xfeat.detectAndComputeDense(image_M)

            idxs_list = self.xfeat.batch_match(out1['descriptors'], out2['descriptors'] )
            m = self.xfeat.refine_matches(out1, out2, matches = idxs_list, batch_idx=0)
            mkpts_F = m[:,0:2].detach()#.cpu().numpy()
            mkpts_M = m[:,2:4].detach()#.cpu().numpy()
            l_idxs = range(m.shape[0])
            return mkpts_F, mkpts_M, l_idxs, None
        
        if method == "xfeats_cv2DMatch":
            mkpts_F, mkpts_M = self.xfeat.match_xfeat(image_F, image_M)
            img_F = tensor2im(image_F)
            img_M = tensor2im(image_M)
            canvas, matches = warp_corners_and_draw_matches(mkpts_F, mkpts_M, img_F, img_M)
            l_idxs = [ m.queryIdx for m in matches ]
            
            return torch.from_numpy(mkpts_F), torch.from_numpy(mkpts_M), l_idxs, canvas
        else:
            raise ValueError(f"method {method} not supported")


    def forward(self, image_F: Tensor_image_NCHW, image_M: Tensor_image_NCHW, method="xfeats_cv2DMatch") -> Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw, List[int], Np_image_HWC]:
        return self.match(image_F, image_M, method)
    

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.5, maxIters=1_000, confidence=0.999):
    """
    Warp corners and draw matches, modified from https://github.com/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb
    """
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, method=method, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches, matches
