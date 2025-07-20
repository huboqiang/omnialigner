import numpy as np
import torch
import dask.array as da

import omnialigner as om
from omnialigner.utils.field_transform import grid_M_to_tfrs, tfrs_to_grid_M, calculate_M_from_theta, calculate_theta_from_M, tfrs_inv
from omnialigner.utils.image_transform import apply_tfrs_to_dask
from omnialigner.plotting import keypoint_viz as viz
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_tfrs, Tensor_cv2_affine_M, Dask_image_HWC

class KeypointPairs(object):
    """A class for managing and manipulating paired keypoints between two images.

    This class handles the storage, transformation, and visualization of keypoint pairs
    between a fixed (reference) image and a moving image. It provides methods for
    calculating transformations, moving keypoints, and visualizing matches.

    Attributes:
        dev (torch.device): Device to use for computations.
        dataset (dict): Dictionary containing image and keypoint data:
            - image_input: Moving image tensor
            - image_label: Fixed image tensor
            - test_input/train_input: Moving image keypoints
            - test_label/train_label: Fixed image keypoints
            - index_matches: Indices of matched keypoint pairs
            - flip: Flip parameters if applicable
            - best_angle: Best rotation angle if applicable
        image_M_adjusted (Tensor_image_NCHW): Transformed moving image
        best_angle (float): Best rotation angle found
        flip (tuple): Flip parameters (x, y) if applicable
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
                 device=torch.device("cpu")):
        """Initialize KeypointPairs object.

        Args:
            image_F (Tensor_image_NCHW, optional): Fixed image tensor. Defaults to None.
            image_M (Tensor_image_NCHW, optional): Moving image tensor. Defaults to None.
            mkpts_F (Tensor_kpts_N_xy_raw, optional): Fixed image keypoints. Defaults to None.
            mkpts_M (Tensor_kpts_N_xy_raw, optional): Moving image keypoints. Defaults to None.
            index_matches (list, optional): Indices of matched keypoint pairs. Defaults to [].
            image_M_adjusted (Tensor_image_NCHW, optional): Transformed moving image. Defaults to None.
            best_angle (float, optional): Best rotation angle. Defaults to None.
            flip (tuple, optional): Flip parameters (x, y). Defaults to None.
            device (torch.device, optional): Computation device. Defaults to CPU.
        """
        self.dev = device
        self.dataset = {"flip": flip, "best_angle": best_angle, "index_matches": index_matches}
        self.image_M_adjusted = image_M_adjusted
        self.best_angle = best_angle
        self.flip = flip
        self.prepare_sample(image_F, image_M, mkpts_F, mkpts_M, index_matches)

    def prepare_sample(self, image_F:Tensor_image_NCHW, image_M:Tensor_image_NCHW, mkpts_F:Tensor_kpts_N_xy_raw, mkpts_M:Tensor_kpts_N_xy_raw, index_matches=None):
        """Prepare image and keypoint data for processing.

        Args:
            image_F (Tensor_image_NCHW): Fixed image tensor.
            image_M (Tensor_image_NCHW): Moving image tensor.
            mkpts_F (Tensor_kpts_N_xy_raw): Fixed image keypoints.
            mkpts_M (Tensor_kpts_N_xy_raw): Moving image keypoints.
            index_matches (list, optional): Indices of matched keypoint pairs. Defaults to None.
        """
        self.dataset['image_input'] = image_M
        self.dataset['image_label'] = image_F
        self.update_kpts(mkpts_F, mkpts_M, index_matches)

    def calculate_kpt_dists(self, indices=None) -> np.ndarray:
        """Calculate distances between matched keypoint pairs.

        Args:
            indices (array-like, optional): Indices of keypoint pairs to calculate distances for.
                If None, uses all pairs. Defaults to None.

        Returns:
            np.ndarray: Array of L1 distances between matched keypoint pairs.
        """
        mkpts_M = self.dataset['test_input']
        mkpts_F = self.dataset['test_label']
        if indices is None:
            indices = np.arange(self.dataset["test_input"].shape[0])

        dist = torch.sum(torch.abs(mkpts_M[indices,:]-mkpts_F[indices,:]), axis=1).detach().cpu().numpy()
        return dist

    def reorder_by_dist(self):
        """Reorder keypoint pairs based on their distances.
        
        Sorts keypoint pairs in ascending order of their L1 distances.
        Updates the dataset with reordered keypoints and adjusted match indices.
        """
        mkpts_M = self.dataset['test_input'].detach()
        mkpts_F = self.dataset['test_label'].detach()
        index_matches = self.dataset["index_matches"]
        dist = self.calculate_kpt_dists()

        reorder_idx = np.argsort(dist)

        index_matches = np.argsort(reorder_idx)[index_matches]
        mkpts_M = mkpts_M[reorder_idx,:]
        mkpts_F = mkpts_F[reorder_idx,:]

        self.update_kpts(mkpts_F, mkpts_M, index_matches)

    def update_new_mkpts_M(self, image_M:Tensor_image_NCHW, mkpts_M:Tensor_kpts_N_xy_raw, reorder_by_dist=False):
        """Update moving image and its keypoints.

        Args:
            image_M (Tensor_image_NCHW): New moving image tensor.
            mkpts_M (Tensor_kpts_N_xy_raw): New moving image keypoints.
            reorder_by_dist (bool, optional): Whether to reorder pairs by distance.
                Defaults to False.
        """
        self.dataset['image_input'] = image_M.to(self.dev)
        index_matches = self.dataset["index_matches"]
        self.dataset['train_input'] = mkpts_M[index_matches, :]
        self.dataset['test_input'] =  mkpts_M

        if reorder_by_dist:
            self.reorder_by_dist()

    def update_kpts(self, mkpts_F:Tensor_kpts_N_xy_raw, mkpts_M:Tensor_kpts_N_xy_raw, index_matches):
        """Update keypoint pairs and match indices.

        Args:
            mkpts_F (Tensor_kpts_N_xy_raw): Fixed image keypoints.
            mkpts_M (Tensor_kpts_N_xy_raw): Moving image keypoints.
            index_matches (array-like): Indices of matched keypoint pairs.
        """
        self.dataset['test_input'] =  mkpts_M
        self.dataset['test_label'] =  mkpts_F
        self.update_indices(index_matches)

    def update_indices(self, index_matches):
        """Update match indices and corresponding keypoint pairs.

        Args:
            index_matches (array-like): New indices of matched keypoint pairs.
        """
        if len(index_matches) == 0:
            return

        self.dataset['train_input'] = self.dataset['test_input'][index_matches, :].clone()
        self.dataset['train_label'] = self.dataset['test_label'][index_matches, :].clone()
        self.dataset["index_matches"] = index_matches

    def __calc_M(self, np1, np2, n):
        """Calculate transformation matrix using least squares.

        Args:
            np1 (torch.Tensor): Target points.
            np2 (torch.Tensor): Source points.
            n (int): Number of point pairs.

        Returns:
            torch.Tensor: 3x3 transformation matrix.
        """
        np2_homogeneous = torch.hstack([np2, torch.ones((n, 1))]).float()
        grid_M_3x3, _, _, _ = torch.linalg.lstsq(np2_homogeneous, np1, rcond=None)
        return grid_M_3x3

    def calculate_grid_M(self, inverse=False):
        """Calculate grid transformation matrix.

        Args:
            inverse (bool, optional): Whether to calculate inverse transformation.
                Defaults to False.

        Returns:
            torch.Tensor: 3x3 grid transformation matrix.
        """
        _, _, h1, w1 = self.dataset['image_label'].shape
        _, _, h2, w2 = self.dataset['image_input'].shape
        np1 = self.dataset["train_label"].float() / torch.FloatTensor([w1/2, h1/2]).float()-1
        np2 = self.dataset["train_input"].float() / torch.FloatTensor([w2/2, h2/2]).float()-1
        n = np2.shape[0]
        if inverse:
            return self.__calc_M(np2, np1, n)
        
        return self.__calc_M(np1, np2, n)

    def calculate_affine_M(self, inverse=False) -> Tensor_cv2_affine_M:
        """Calculate affine transformation matrix.

        Args:
            inverse (bool, optional): Whether to calculate inverse transformation.
                Defaults to False.

        Returns:
            Tensor_cv2_affine_M: 2x3 affine transformation matrix.
        """
        _, _, h1, w1 = self.dataset['image_label'].shape
        grid_M_3x3 = self.calculate_grid_M(inverse=inverse)
        M_ = calculate_M_from_theta(grid_M_3x3.T, h1, w1).numpy()[0:2, :]
        return M_

    def calculate_tfrs(self) -> Tensor_tfrs:
        """Calculate transformation parameters.

        Returns:
            Tensor_tfrs: Transformation parameters [angle, tx, ty, sx, sy].
        """
        img_M = self.dataset["image_input"]
        M_ = self.calculate_affine_M(inverse=True)
        tensor_affineM = calculate_theta_from_M(torch.from_numpy(M_).float(), img_M.shape[2], img_M.shape[3])
        tfrs = grid_M_to_tfrs(tensor_affineM)
        return tfrs

    def move_kpt_M(self, tensor_tfrs:Tensor_tfrs=None) -> Tensor_kpts_N_xy_raw:
        """Transform moving image keypoints using current transformation.

        Returns:
            Tensor_kpts_N_xy_raw: Transformed keypoint coordinates.
        """
        _, _, h2, w2 = self.dataset['image_input'].shape
        if tensor_tfrs is None:
            grid_M_3x3 = self.calculate_grid_M()
        else:
            grid_M_3x3 = tfrs_inv(tensor_tfrs)[0:2].T

        print("grid_M_3x3", grid_M_3x3)
        np2 = self.dataset["train_input"].float() / torch.FloatTensor([w2/2, h2/2]).float()-1
        n = np2.shape[0]
        np2_homogeneous = torch.hstack([np2, torch.ones((n, 1))]).float()
        np_2_ = ((np2_homogeneous @ grid_M_3x3)+1)*torch.FloatTensor([w2/2, h2/2]).float()-1
        return np_2_

    def move_img_M_cv2(self) -> np.ndarray:
        """Transform moving image using OpenCV affine transformation.

        Returns:
            np.ndarray: Transformed image array.
        """
        import cv2
        img_M = self.dataset["image_input"]
        da_M = om.tl.tensor2im(img_M)
        M_ = self.calculate_affine_M(inverse=False)
        _, _, h1, w1 = self.dataset['image_label'].shape
        da_M_moved = cv2.warpAffine(da_M, M_, (h1, w1))
        return da_M_moved

    def move_img_M(self, tensor_tfrs:Tensor_tfrs=None) -> Dask_image_HWC:
        """Transform moving image using calculated or provided transformation.

        Args:
            tensor_tfrs (Tensor_tfrs, optional): Transformation parameters.
                If None, calculates from keypoint pairs. Defaults to None.

        Returns:
            Dask_image_HWC: Transformed image array.
        """
        img_M = self.dataset["image_input"]
        if tensor_tfrs is None:
            tensor_tfrs = self.calculate_tfrs()

        da_M = da.from_array(om.tl.tensor2im(img_M))
        da_M_moved = apply_tfrs_to_dask(da_M, tensor_tfrs=[tensor_tfrs], tile_size=(100, 100, 1))
        return da_M_moved

    def plot_dist_distribute(self, ax=None):
        """Plot distribution of keypoint pair distances.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.

        Returns:
            matplotlib.axes.Axes: Plot axes.
        """
        return viz.plot_kg_dist_distribute(self.dataset, ax)

    def plot_dataset(self, **kwargs):
        """Plot keypoint pairs and their matches.

        Args:
            **kwargs: Additional arguments passed to visualization function.

        Returns:
            matplotlib.figure.Figure: Plot figure.
        """
        return viz.plot_kg_dataset(self.dataset, **kwargs)

    def plot_diff_gradient(self, show_grad=True, n_grids=10):
        """Plot gradient of keypoint differences.

        Args:
            show_grad (bool, optional): Whether to show gradient arrows. Defaults to True.
            n_grids (int, optional): Number of grid divisions. Defaults to 10.

        Returns:
            tuple: Plot figure and axes.
        """
        return viz.plot_kpts_diff_gradient(self.dataset, grid_size=n_grids, show_grad=show_grad)
