from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm
from cellpose.denoise import CellposeDenoiseModel
from cellpose.utils import outlines_list

from omnialigner.keypoints.keypoint_pairs import KeypointPairs
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_tfrs, Tensor_cv2_affine_M, Dask_image_HWC
from omnialigner.plotting import keypoint_viz as viz



class MatchedCells(KeypointPairs):
    """A class for matching and analyzing cell pairs between two images.

    This class extends KeypointPairs to provide specialized functionality for
    cell detection, matching, and analysis. It uses CellPose for cell detection
    and implements methods for finding corresponding cells between images based
    on keypoint matches.

    Attributes:
        knn_cell_match (int): Number of nearest neighbors to consider for cell matching.
        knn_matched_threshold (float): Threshold for considering cells as matched.
        dist_threshold (float): Distance threshold for filtering matched cells.
        All attributes from KeypointPairs are inherited.
    """

    def __init__(self, 
                 image_F:Tensor_image_NCHW=None,
                 image_M:Tensor_image_NCHW=None, 
                 mkpts_F:Tensor_kpts_N_xy_raw=None, 
                 mkpts_M:Tensor_kpts_N_xy_raw=None, 
                 index_matches=[], 
                 image_M_adjusted=None, 
                 best_angle=None, 
                 flip=None, 
                 knn_cell_match=5,
                 knn_matched_threshold=0.8,
                 dist_threshold=95,
                 device=torch.device("cpu")):
        """Initialize MatchedCells object.

        Args:
            image_F (Tensor_image_NCHW, optional): Fixed image tensor. Defaults to None.
            image_M (Tensor_image_NCHW, optional): Moving image tensor. Defaults to None.
            mkpts_F (Tensor_kpts_N_xy_raw, optional): Fixed image keypoints. Defaults to None.
            mkpts_M (Tensor_kpts_N_xy_raw, optional): Moving image keypoints. Defaults to None.
            index_matches (list, optional): Indices of matched keypoint pairs. Defaults to [].
            image_M_adjusted (Tensor_image_NCHW, optional): Transformed moving image. Defaults to None.
            best_angle (float, optional): Best rotation angle. Defaults to None.
            flip (tuple, optional): Flip parameters (x, y). Defaults to None.
            knn_cell_match (int, optional): Number of nearest neighbors for cell matching. Defaults to 5.
            knn_matched_threshold (float, optional): Threshold for cell matching. Defaults to 0.8.
            dist_threshold (float, optional): Distance threshold for filtering. Defaults to 95.
            device (torch.device, optional): Computation device. Defaults to CPU.
        """
        super().__init__(
                image_F=image_F,
                image_M=image_M, 
                mkpts_F=mkpts_F, 
                mkpts_M=mkpts_M,
                index_matches=index_matches, 
                image_M_adjusted=image_M_adjusted, 
                best_angle=best_angle,
                flip=flip, 
                device=device)
        self.knn_cell_match = knn_cell_match
        self.knn_matched_threshold = knn_matched_threshold
        self.dist_threshold = dist_threshold
        
    def detect_cells(self, masks: Tuple[np.ndarray, np.ndarray]=None):
        """Detect cells in both fixed and moving images.

        Uses CellPose model for cell detection if masks are not provided.
        Updates the dataset with cell masks and cell coordinates.

        Args:
            masks (tuple, optional): Pre-computed cell masks for both images.
                If None, performs cell detection. Defaults to None.
        """
        if masks is None:
            model_denoise = CellposeDenoiseModel(gpu=False, model_type="nuclei",
                                     restore_type="denoise_nuclei")

            l_imgs = [  self.dataset["image_label"][0,0], self.dataset["image_input"][0,0] ]
            masks, flows, styles, imgs_dn = model_denoise.eval(l_imgs, diameter=4, channels=[0,0])
        
        self.dataset["cell_label"] = masks[0]
        self.dataset["cell_input"] = masks[1]

        self.dataset["cell_coord_label"] = outlines_list(masks[0])
        self.dataset["cell_coord_input"] = outlines_list(masks[1])

    def find_cell_nearest_kpt(self, np_cell_coord: np.ndarray, np_kpt_coords: np.ndarray, k=5):
        """Find k-nearest keypoints for each cell.

        Args:
            np_cell_coord (np.ndarray): Cell coordinates [N, 2].
            np_kpt_coords (np.ndarray): Keypoint coordinates [M, 2].
            k (int, optional): Number of nearest neighbors. Defaults to 5.

        Returns:
            tuple:
                - np.ndarray: Displacement vectors to k-nearest neighbors [N, k, 2].
                - np.ndarray: Indices of k-nearest neighbors [N, k].
        """
        import faiss
        index = faiss.IndexFlatL2(2)
        index.add(np_kpt_coords.astype(np.float32))  # Ensure float32
        _, indices = index.search(x=np_cell_coord.astype(np.float32).copy(), k=k)
        l_D = []
        for i_cell in range(np_cell_coord.shape[0]):
            np_xy_cell = np_cell_coord[i_cell]
            D = np_xy_cell - np_kpt_coords[indices[i_cell]]
            l_D.append(D)

        return np.array(l_D), indices

    def find_match_cell(self):
        """Find matching cells between fixed and moving images.

        Uses k-nearest neighbor keypoints to establish cell correspondences.
        Updates the dataset with matched cell indices and matching scores.
        """
        np_cell_coord_F = np.array([ x.mean(0) for x in self.dataset["cell_coord_label"] ])
        np_kpt_coord_F = self.dataset["train_label"].numpy()
        np_cell_coord_M = np.array([ x.mean(0) for x in self.dataset["cell_coord_input"] ])
        np_kpt_coord_M = self.dataset["train_input"].numpy()

        np_dist_2D_F, cell_knn_kpt_F = self.find_cell_nearest_kpt(np_cell_coord_F, np_kpt_coord_F, self.knn_cell_match)
        np_dist_2D_M, cell_knn_kpt_M = self.find_cell_nearest_kpt(np_cell_coord_M, np_kpt_coord_M, self.knn_cell_match)

        sets_F = [set(cell_knn_kpt_F[i]) for i in range(len(cell_knn_kpt_F))]
        sets_M = [set(cell_knn_kpt_M[i]) for i in range(len(cell_knn_kpt_M))]

        matched_M = set()
        matched_pairs = []

        for i_F, sF in tqdm(enumerate(sets_F)):
            best_i_M = None
            best_score = float('inf')
            for i_M, sM in enumerate(sets_M):
                if i_M in matched_M:
                    continue
                inter = sF & sM
                if len(inter) / self.knn_cell_match >= self.knn_matched_threshold:
                    l_ovlp = list(inter)
                    idx_F = [list(cell_knn_kpt_F[i_F]).index(k) for k in l_ovlp]
                    idx_M = [list(cell_knn_kpt_M[i_M]).index(k) for k in l_ovlp]
                    # Calculate relative displacement differences
                    delta = np_dist_2D_F[i_F, idx_F] - np_dist_2D_M[i_M, idx_M]
                    score = np.abs(delta).mean()

                    if score < best_score:
                        best_score = score
                        best_i_M = i_M
            if best_i_M is not None:
                matched_pairs.append((i_F, best_i_M, best_score))
                matched_M.add(best_i_M)

        self.dataset["cell_matched_label"] = np.array([i for i, _, _ in matched_pairs])
        self.dataset["cell_matched_input"] = np.array([j for _, j, _ in matched_pairs])
        self.dataset["cell_matched_score_dist"] = np.array([k for _, _, k in matched_pairs])

    def export_cell_dataset(self):
        """Export matched cell data as a new dataset.

        Creates a new dataset containing only the matched cells that pass
        the distance threshold filter. Updates self.cell_dataset.
        """
        tensor_F = self.dataset['image_label']
        tensor_M = self.dataset["image_input"]

        np_cell_coord_F = np.array([ x.mean(0) for x in self.dataset["cell_coord_label"] ])
        np_cell_coord_M = np.array([ x.mean(0) for x in self.dataset["cell_coord_input"] ])
        np_test_label = torch.from_numpy(np_cell_coord_F)[self.dataset["cell_matched_label"]]
        np_test_input = torch.from_numpy(np_cell_coord_M)[self.dataset["cell_matched_input"]]
        np_dist = self.dataset["cell_matched_score_dist"]
        index_matches = np.array(list(range(np_test_input.shape[0])))[ np_dist <= np.percentile(np_dist, self.dist_threshold)]
        self.cell_dataset = {
            "image_label": tensor_F,
            "image_input": tensor_M,
            "index_matches" : index_matches,
            "test_label": np_test_label,
            "test_input": np_test_input,
            "train_label": np_test_label[index_matches],
            "train_input": np_test_input[index_matches],
        }

    def plot_cell_dataset(self, **kwargs):
        """Plot matched cell pairs.

        Args:
            **kwargs: Additional arguments passed to visualization function.

        Returns:
            matplotlib.figure.Figure: Plot figure showing matched cell pairs.
        """
        self.export_cell_dataset()
        return viz.plot_kg_dataset(self.cell_dataset, **kwargs)
