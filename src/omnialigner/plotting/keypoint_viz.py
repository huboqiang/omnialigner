from typing import Dict, List, Tuple
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from omnialigner.plotting.matplotlib_init import plt
from omnialigner.utils.image_viz import tensor2im
from omnialigner.dtypes import Tensor_image_NCHW, Tensor_kpts_N_xy_raw, Tensor_l_kpt_pair, Tensor_kpt_pair

def plot_kpt_pairs(image: Tensor_image_NCHW, 
               kpts0: Tensor_kpts_N_xy_raw=None, 
               kpts1: Tensor_kpts_N_xy_raw=None, 
               title: str=None,
               c0: str="b",
               c1: str="r",
               s0: float=0.2,
               s1: float=0.2,
               ax: plt.Axes=None) -> plt.Figure:
    """Plot a single frame with keypoints.

    Args:
        image: Image tensor or numpy array in NCHW format.
        kpts0: Optional keypoints of the current frame.
        kpts1: Optional keypoints of the next frame.
        title: Optional title of the plot.
        c0: Color for first set of keypoints (default: "b").
        c1: Color for second set of keypoints (default: "r").
        s0: Size of first set of keypoints (default: 0.2).
        s1: Size of second set of keypoints (default: 0.2).
        ax: Optional matplotlib axis to plot on.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    ax.imshow(tensor2im(image) if torch.is_tensor(image) else image)
    if kpts0 is not None and kpts0.shape[0] > 0:
        ax.scatter(kpts0[:, 0], kpts0[:, 1], c=c0, s=s0)
    if kpts1 is not None and kpts1.shape[0] > 0:
        ax.scatter(kpts1[:, 0], kpts1[:, 1], c=c1, s=s1)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.close()
    return fig

def plot_align_batch(
        transformed_images: Tensor_image_NCHW, 
        l_kpt_pairs: Tensor_l_kpt_pair = None, 
        batch_indices: List[int] = [0],
        file_png: str = None, 
        n_cols: int=8,
        show_keypoints: bool = False,
        figsize: Tuple[int, int] = (20, 12)
    ) -> plt.Figure:
    fig = plt.figure(figsize=figsize)

    total_images, _, H, W = transformed_images.shape
    mean_diff = None
    if l_kpt_pairs is not None and len(l_kpt_pairs) > 0:
        mean_diff = [ torch.mean(torch.abs(k[0][1]-k[0][0]), 0) for k in l_kpt_pairs ]
    
    np_image = transformed_images.cpu().detach().numpy()
    idx_x = int(np.ceil(total_images / n_cols))
    for i_image in range(total_images):
        ax = fig.add_subplot(idx_x, n_cols, i_image+1)
        ax.imshow(np_image[i_image, 0, :, :])
        if show_keypoints:
            _plot_keypoints(l_kpt_pairs, i_image, ax, W, H)

        ax.set_xlim(0, np_image.shape[2])
        ax.set_ylim(0, np_image.shape[3])
        image_index = batch_indices[0].item()+i_image
        kpt_diff = np.nan
        if mean_diff is not None and i_image < len(mean_diff) and len(l_kpt_pairs[i_image]) > 0:
            kpt_diff = torch.mean(mean_diff[i_image]).item()
        ax.set_title(f"{image_index}, {kpt_diff:.3f}")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.invert_yaxis()
    
    if file_png is not None:
        fig.savefig(file_png)
        plt.close(fig)

    return fig



def plot_ovlp_kpts(
        l_imgs: List[List[Tensor_image_NCHW]], 
        l_kpt_pairs: List[Tensor_kpts_N_xy_raw|Tensor_kpt_pair], 
        scale_level: float=1., 
        file_png: str=None, 
        **kwargs
    ) -> plt.Figure:
    figsize = kwargs.get("figsize", (10, 10))
    n_cols = kwargs.get("n_cols", 8)
    n_rows = kwargs.get("n_rows", len(l_imgs) // n_cols + 1)
    fig = plt.figure(figsize=figsize)
    for i_layer, (img, kpt_pair) in enumerate(zip(l_imgs, l_kpt_pairs)):
        if len(kpt_pair) == 2:
            kpt_F = kpt_pair[0]
            kpt_M = kpt_pair[1]
            img_F = tensor2im(img[0])
            img_M = tensor2im(img[1])
        else:
            if i_layer+1 >= len(l_imgs) or i_layer+1 >= len(l_kpt_pairs):
                continue
        
            kpt_F = l_kpt_pairs[i_layer][0]
            kpt_M = l_kpt_pairs[i_layer+1][0]
            img_F = tensor2im(l_imgs[i_layer][0])
            img_M = tensor2im(l_imgs[i_layer+1][0])
        
        ax = fig.add_subplot(n_rows, n_cols, i_layer+1)
        da_out = 255-img_F
        da_out[:, :, 1] = 0
        da_out[:, :, 0] = 255-img_M[:, :, 0]
        ax.imshow(da_out)
        np_kptsF = kpt_F.cpu().numpy() / scale_level * img_F.shape[1]
        np_kptsM = kpt_M.cpu().numpy() / scale_level * img_M.shape[1]
        ax.scatter(np_kptsF[:, 0], np_kptsF[:, 1], c="cyan", s=15)
        ax.set_title(f"{i_layer}")
        ax.scatter(np_kptsM[:, 0], np_kptsM[:, 1], c="yellow", s=5)

    if file_png is not None:
        fig.savefig(file_png)
        plt.close(fig)

    return fig


def _plot_keypoints(l_kpt_pairs, i_image, ax, W, H):
    if l_kpt_pairs is None or len(l_kpt_pairs) == 0:
        return
    
    if len(l_kpt_pairs) <= i_image:
        return
    
    kpts = None
    if i_image < len(l_kpt_pairs) and len(l_kpt_pairs[i_image]) > 0:
        kpts = l_kpt_pairs[i_image][0][0].detach().cpu().numpy()
    elif len(l_kpt_pairs[-1]) > 0:
        kpts = l_kpt_pairs[-1][0][1].detach().cpu().numpy()

    if kpts is not None:
        kpts = kpts*np.array([W, H])
        ax.scatter(kpts[:, 0], kpts[:, 1], c="b", s=0.2)


def plot_2d_overlap(batch_images: Tensor_image_NCHW, 
                    l_kpt_pairs: List[Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]], 
                    file_png: str, 
                    grid2d_model: nn.Module, # Grid2DModelDual
                    tensor_size: List[int]=[256, 256]
                ) -> plt.Figure:
    """
        batch_images:       [2, C, H, W] tensor images
        l_kpt_pairs:        [kpts_F[:], kpts_M[:]]
        file_png:           file name for results
        grid2d_model:       A type of Grid2DModelDual model trained by align_2d
        tensor_size:        [256, 256]
    """
    tensor_F = batch_images[0:1]
    tensor_M = batch_images[1:2]
    kpts_F = l_kpt_pairs[0][0][0]
    kpts_M = l_kpt_pairs[0][1][0]
    
    tensor_M_to_F, kpts_M_to_F = grid2d_model.forward(tensor_M, kpts_M)
    tensor_F_to_M, kpts_F_to_M = grid2d_model.backforward(tensor_F, kpts_F)
    
    np_image_F = tensor2im(tensor_F.cpu())
    np_image_F[:, :, 1] = 0
    if tensor_M_to_F is not None:
        H_, W_ = np_image_F.shape[0], np_image_F.shape[1]
        tensor_M_to_F = F.interpolate(tensor_M_to_F, size=[H_, W_], mode="bilinear")
        np_image_F[:, :, 2] = tensor2im(tensor_M_to_F.cpu())[:, :, 0]
    np_kpt_F = tensor_size*kpts_F.cpu().detach().numpy()
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(np_image_F)
    ax.plot(np_kpt_F[:, 0], np_kpt_F[:, 1], "bo")
    if kpts_M_to_F is not None:
        np_kpt_M_to_F = tensor_size*kpts_M_to_F.cpu().detach().numpy()
        ax.plot(np_kpt_M_to_F[:, 0], np_kpt_M_to_F[:, 1], "r.")
    
    np_image_M = tensor2im(tensor_M.cpu())
    np_image_M[:, :, 1] = 0
    if tensor_F_to_M is not None:
        np_image_M[:, :, 2] = tensor2im(tensor_F_to_M.cpu())[:, :, 0]
    
    np_kpt_M = tensor_size*kpts_M.cpu().detach().numpy()
    
    ax = fig.add_subplot(1,2,2)
    ax.imshow(np_image_M)
    ax.plot(np_kpt_M[:, 0], np_kpt_M[:, 1], "bo")
    if kpts_F_to_M is not None:
        np_kpt_F_to_M = tensor_size*kpts_F_to_M.cpu().detach().numpy()
        ax.plot(np_kpt_F_to_M[:, 0], np_kpt_F_to_M[:, 1], "r.")
    
    fig.savefig(file_png)
    plt.close(fig)
    return fig


def plot_2d_overlap_ANHIR_rtre(image_M, np_landmark_M, image_F, np_landmark_F, image_M_to_F, np_landmark_M_to_F, rtre_final=np.array([np.nan])):
    H, W = image_F.shape[0], image_F.shape[1]
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1,3,1)
    ax.imshow(image_M)
    ax.plot(np_landmark_M[:, 0], np_landmark_M[:, 1], "r.")
    ax.set_title("source image")

    ax = fig.add_subplot(1,3,2)
    ax.imshow(image_F)
    ax.set_title("target image")
    if np_landmark_F is not None:
        ax.plot(np_landmark_F[:, 0], np_landmark_F[:, 1], ".")

    ax = fig.add_subplot(1,3,3)
    np_out = image_F.copy()
    np_out[:, :, 0] = tensor2im(image_M_to_F)[:, :, 0]
    np_out[:, :, 1] = 0
    ax.imshow(np_out)
    ax.plot(np_landmark_M_to_F[:, 0], np_landmark_M_to_F[:, 1], "r.")
    str_rtres = f"overlap image"
    if np_landmark_F is not None:
        ax.plot(np_landmark_F[:, 0], np_landmark_F[:, 1], "b.")
        str_rtres = f"overlap image, rtre={np.median(rtre_final):.2e}"

    ax.set_title(str_rtres)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    plt.close()
    return fig



def _plot_kpts_selected(pt_kpts: Tensor_kpts_N_xy_raw, 
                        l_idxs: List[int]=None, 
                        show_all_kps: bool=True, 
                        ax: plt.Axes=None,
                        color="r",
                        **kwargs
                    ) -> plt.Figure:
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    color_used = kwargs.get("color_used", "g")
    size_all = kwargs.get("size_all", 1)
    size_used = kwargs.get("size_used", 5)
    
    mkpts = pt_kpts.cpu().detach().numpy()
    if show_all_kps:
        ax.scatter(mkpts[:,0], mkpts[:,1], c=color, s=size_all)

    if l_idxs is not None:
        ax.scatter(mkpts[l_idxs,0], mkpts[l_idxs,1], c=color_used, s=size_used)

    plt.close()
    return fig


def plot_kpts_diff(pt_kpts_0: Tensor_kpts_N_xy_raw, 
                    pt_kpts_1: Tensor_kpts_N_xy_raw, 
                    l_idxs: List[int]=None, 
                    ax: plt.Axes=None,
                    **kwargs
                ) -> plt.Figure:
    mkpts_0 = pt_kpts_0.cpu().detach().numpy()
    mkpts_1 = pt_kpts_1.cpu().detach().numpy()
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    
    color_all = kwargs.get("color_all", "b")
    color_used = kwargs.get("color_used", "r")
    size_all = kwargs.get("size_all", 1)
    size_used = kwargs.get("size_used", 1)
    head_width = kwargs.get("head_width", mkpts_0.max()/500.)
    head_length = kwargs.get("head_length", mkpts_0.max()/500.)
    linewidth = kwargs.get("linewidth", 1)
    ax.scatter(mkpts_0[:, 0], mkpts_0[:, 1], c=color_all, s=size_all)
    ax.scatter(mkpts_1[:, 0], mkpts_1[:, 1], c=color_used, s=size_used)
    for i in l_idxs:
        # ax.plot([mkpts_1[i,0], mkpts_0[i,0]], [mkpts_1[i,1], mkpts_0[i,1]], "g-")
        ax.arrow(mkpts_1[i, 0], mkpts_1[i, 1], mkpts_0[i, 0] - mkpts_1[i, 0], mkpts_0[i, 1] - mkpts_1[i, 1], head_width=head_width, head_length=head_length, fc='green', ec='green', linewidth=linewidth)

    ax.invert_yaxis()
    plt.close()
    return fig


def plot_kpts_gradient(pt_kpts_F: Tensor_kpts_N_xy_raw, 
                    pt_kpts_M: Tensor_kpts_N_xy_raw, 
                    l_idxs: List[int]=None, 
                    img_size: List[int]=[1600, 1600], 
                    grid_size: int=10, 
                    ax: plt.Axes=None) -> plt.Figure:
    """Plot keypoint gradients between two sets of points.

    Args:
        pt_kpts_F: First set of keypoints (Nx2).
        pt_kpts_M: Second set of keypoints (Nx2).
        l_idxs: Optional list of indices to highlight specific keypoints.
        img_size: Image size as [height, width] (default: [1600, 1600]).
        grid_size: Size of the gradient grid (default: 10).
        ax: Optional matplotlib axis to plot on.

    Returns:
        matplotlib.figure.Figure: The generated plot figure showing keypoint gradients.
    """
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    
    mkpts_F = pt_kpts_F.cpu().detach().numpy()
    mkpts_M = pt_kpts_M.cpu().detach().numpy()
    # ax.scatter(mkpts_F[l_idxs, 0], mkpts_F[l_idxs, 1], c="b", s=10)
    # ax.scatter(mkpts_M[l_idxs, 0], mkpts_M[l_idxs, 1], c="r", s=10)

    grid_x = np.linspace(0, img_size[0], grid_size + 1)
    grid_y = np.linspace(0, img_size[1], grid_size + 1)

    grid_avg_diff = np.zeros((grid_size, grid_size, 2))
    mkpts_F = mkpts_F[l_idxs, :]
    mkpts_M = mkpts_M[l_idxs, :]
    # Calculate average differences in each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            x_min, x_max = grid_x[i], grid_x[i + 1]
            y_min, y_max = grid_y[j], grid_y[j + 1]

            in_cell = np.where((mkpts_F[:, 0] >= x_min) & (mkpts_F[:, 0] < x_max) & 
                               (mkpts_F[:, 1] >= y_min) & (mkpts_F[:, 1] < y_max))[0]
            
            if len(in_cell) > 0:
                diffs = mkpts_F[in_cell] - mkpts_M[in_cell]
                avg_diff = np.mean(diffs, axis=0)
                grid_avg_diff[i, j] = avg_diff

                # Plot the average difference as a vector
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                arrow_width = np.log10(len(in_cell)+1)

                ax.scatter(center_x, center_y, s=15, c="k")
                ax.arrow(center_x- avg_diff[0], center_y - avg_diff[1], avg_diff[0], avg_diff[1], 
                         head_width=5, head_length=8, fc='c', ec='c', linewidth=arrow_width
                         )
                

    # plt.close()
    return fig, grid_avg_diff

def plot_kpts_diff_gradient(dataset, grid_size=10, show_grad=True):
    img_src = tensor2im(dataset["image_input"])
    img_tar = tensor2im(dataset["image_label"])
    H, W = img_src.shape[0], img_src.shape[1]
    np_out = np.zeros([H, W, 3], dtype=np.uint8)+255
    np_out[:,:,0] = img_src[:,:,0]  # red move
    np_out[:,:,2] = img_tar[:,:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(np_out)
    if show_grad:
        mkpt_src = dataset["test_input"]
        mkpt_tar = dataset["test_label"]
        _, grid_avg_diff = plot_kpts_gradient(mkpt_tar, mkpt_src, dataset["index_matches"], img_size=[H, W], grid_size=grid_size, ax=ax)
        return grid_avg_diff

    return None


def plot_kg_dist_distribute(dataset, ax=None):
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    
    index_matches = np.array(dataset["index_matches"])
    mkpts_M = dataset['test_input'].detach().numpy()
    mkpts_F = dataset['test_label'].detach().numpy()
    ax.plot( np.sum(np.abs(mkpts_M-mkpts_F), axis=1) )
    ax.plot( index_matches, np.sum(np.abs(mkpts_M[index_matches]-mkpts_F[index_matches]), axis=1), "." )
    return fig


def plot_kg_dataset(dataset: Dict[str, str], **kwargs) -> plt.Figure:
    """
    plot keypoint pairs in dataset.
    Args:
        dataset: KeypointPairs.dataset
    """
    tensor_F = dataset['image_label']
    tensor_M = dataset["image_input"]    
    idxs = dataset["index_matches"]

    fig = plt.figure(figsize=(18,6))

    ax = fig.add_subplot(1,3,1)
    _plot_kpts_selected(dataset["test_label"], l_idxs=idxs, ax=ax, color="b", **kwargs)
    ax.set_title("image_fixed")
    if tensor_F is not None:
        image_F = tensor2im(tensor_F)
        ax.imshow(image_F)
        ax.set_ylim(0, image_F.shape[0])
        ax.set_xlim(0, image_F.shape[1])
        ax.invert_yaxis()


    ax = fig.add_subplot(1,3,2)
    _plot_kpts_selected(dataset["test_input"], l_idxs=idxs, ax=ax, color="r", **kwargs)
    ax.set_title("image_move")
    if tensor_M is not None:
        image_M = tensor2im(tensor_M)
        ax.imshow(image_M)
        ax.set_ylim(0, image_M.shape[0])
        ax.set_xlim(0, image_M.shape[1])
        ax.invert_yaxis()

    ax = fig.add_subplot(1,3,3)
    plot_kpts_diff(dataset["test_label"], dataset["test_input"], idxs, ax=ax, **kwargs)
    ax.set_title("kpt movement, r(M)->b(F)")
    plt.close()
    return fig


def viz_grid_kan(im0, im1_v2, dataset, mkpts_pred, model_type="kan"):
    np_out = np.zeros([im0.shape[0], im0.shape[0], 3], dtype=np.uint8)
    np_out[:, :, 0] = im0[:, :, 0]
    np_out[:, :, 2] = im1_v2[:, :, 0]
    

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(np.hstack([np_out]))
    ax.plot(dataset['train_label'][:,0], dataset['train_label'][:,1], "co", label="src")
    ax.plot(mkpts_pred[:,0], mkpts_pred[:,1], "go", label="trans")

    err = np.mean( np.abs(dataset['train_label'].numpy() - mkpts_pred) )
    ax.set_title(f"{model_type}, mean_dist = {err:.2f}")
    ax.legend()
    plt.close()
    return fig
