from typing import List, Tuple
import numpy as np
import dask.array as da
from tqdm import tqdm

import omnialigner as om
from omnialigner.dtypes import Tensor_image_NCHW, Dask_image_NCHW, Tensor_kpts_N_xy_raw, Np_image_HWC
from omnialigner.plotting.matplotlib_init import plt
from omnialigner.utils.image_viz import tensor2im
from omnialigner.plotting.keypoint_viz import plot_kpt_pairs
from omnialigner.omni_3D import Omni3D


def plot_nchw_2d(
        om_data: Omni3D,
        aligned_tag="PAD",
        l_layers: List[int] = None,
        l_kpt_pairs: List[Tuple[Tensor_kpts_N_xy_raw, Tensor_kpts_N_xy_raw]] = None
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
    fig = plt.figure(figsize=om_data.plt_figsize)
    n_col, n_row = om_data.plt_row_col
    if l_layers is None:
        l_layers = list(range(len(om_data)))

    aligned_tensor = om_data.load_3d_NCHW(aligned_tag, l_layers=l_layers)
    for i_layer in l_layers:
        ax = fig.add_subplot(n_col, n_row, i_layer+1)
        if isinstance(aligned_tensor, Tensor_image_NCHW):
            img = tensor2im(aligned_tensor[i_layer:i_layer+1])
        elif isinstance(aligned_tensor, Dask_image_NCHW):
            img = da.moveaxis(aligned_tensor[i_layer], 0, -1).compute()
        else:
            raise ValueError(f"Unsupported type: {type(aligned_tensor)}")

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        kwargs = {
            "image": img,
            "title": f"{i_layer}",
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


def generte_3d_rgba(om_data: Omni3D, np_masks=None) -> List[Np_image_HWC]:
    """Generate RGBA representation for 3D visualization.

    Args:
        om_data: Omni3D dataset object.
        np_masks: Optional list of masks for each layer.

    Returns:
        List[numpy.ndarray]: List of RGBA images for each layer.
    """
    l_img_rgba = []
    for i_layer in tqdm(range(len(om_data))):
        is_HE = (om_data.proj_info.get_dtype(i_layer=i_layer) == "HE")
        tag = "raw" if is_HE else "gray"
        da_img = om.align.apply_nonrigid_HD(
            om_data,
            i_layer=i_layer,
            tag=tag,
            overwrite_cache=True,
            unpad=False
        ).compute()
        if np_masks is None:
            mask = om.align.apply_nonrigid_HD(
                om_data,
                i_layer=i_layer,
                tag="mask",
                overwrite_cache=True,
                unpad=False
            )[:, :, 0].compute() < 0.5
        else:
            mask = np_masks[i_layer] < 0.5
        
        h, w, _ = da_img.shape
        img_rgba = np.zeros([h, w, 4], dtype=np.uint8)

        da_img[mask > 0, :] = 255
        img_rgba[:, :, 0:3] = da_img
        img_rgba[:, :, 3] = 255*(1-mask)
        l_img_rgba.append(img_rgba)
    return l_img_rgba
