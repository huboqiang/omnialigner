import pandas as pd
import torch
import numpy as np
import dask.array as da
import matplotlib.colors as mcolors
from omnialigner.dtypes import Tensor_image_NCHW, Np_image_HWC, Dask_image_HWC

def tensor2im(tensor: Tensor_image_NCHW, is_uint8=True)-> Np_image_HWC:
    """Convert a PyTorch tensor to a numpy image array.

    Args:
        tensor (Tensor_image_NCHW): Input tensor in NCHW format, scaled to [0, 1].
        is_uint8 (bool, optional): Whether to convert to uint8 format. Defaults to True.

    Returns:
        Np_image_HWC: Numpy array in HWC format. If is_uint8 is True, values are scaled
            to [0, 255] and converted to uint8.
    """
    if is_uint8:
        tensor1 = 255.*(tensor)
        image = tensor1.detach().permute(0,2,3,1)[0].numpy().astype(np.uint8)
        return image
    else:
        return tensor.detach().permute(0,2,3,1)[0].numpy()

def im2tensor(image: Np_image_HWC| Dask_image_HWC, is_uint8=True)-> Tensor_image_NCHW:
    """Convert a numpy/dask image array to a PyTorch tensor.

    Args:
        image (Np_image_HWC | Dask_image_HWC): Input image in HWC format.
            If it's a Dask array, it will be computed before conversion.
        is_uint8 (bool, optional): Whether input is in uint8 format (0-255).
            If True, values are scaled to [0, 1]. Defaults to True.

    Returns:
        Tensor_image_NCHW: PyTorch tensor in NCHW format, scaled to [0, 1].
    """
    if isinstance(image, Dask_image_HWC):
        image = image.compute()
    
    image1 = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).permute(0,3,1,2)
    if is_uint8:
        return image1 / 255.
    
    return image1

def rgb_he_to_gray(da_image: da.Array, is_HE: bool, alpha_he: int=3, alpha_ihc: int=5, **kwargs)-> da.Array:
    """Convert RGB image to grayscale with specific processing for HE/IHC images.

    Args:
        da_image (da.Array): Input dask array in HWC format.
        is_HE (bool): Whether the image is H&E stained.
        alpha_he (int, optional): Weight for HE channel enhancement. Defaults to 3.
        alpha_ihc (int, optional): Weight for IHC channel enhancement. Defaults to 5.
        **kwargs: Additional arguments (unused).

    Returns:
        da.Array: Grayscale image as dask array [H, W, 1], uint8 type.
            For HE images: Inverted mean of RGB channels * alpha_he
            For IHC images: Last channel * alpha_ihc
    """
    if is_HE:
        da_he = 255 - da_image.mean(axis=2)
        da_img = da.clip(alpha_he * da_he[:, :, None], 0, 255)
    else:
        da_ihc = da_image[:, :, -1]
        da_img = da.clip(alpha_ihc * da_ihc[:, :, None], 0, 255)
    
    return da_img.astype(np.uint8)


def get_bg_tensor(N: int=1, is_gray: bool=False, is_bright: bool=True) -> Tensor_image_NCHW:
    """Generate background tensor for image padding.

    Args:
        N (int, optional): Number of images in batch. Defaults to 1.
        is_gray (bool, optional): Whether to use grayscale background.
            Always returns black (0) if True. Defaults to False.
        is_bright (bool, optional): Whether to use bright background.
            Only used if is_gray is False. Defaults to True.

    Returns:
        Tensor_image_NCHW: Background tensor of shape [N, 3, 1, 1].
            Values are either 0 (black) or 1 (white) based on parameters.
    """
    if is_gray:
        return torch.repeat_interleave(torch.FloatTensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1), N, dim=0)
    
    if is_bright:
        return torch.repeat_interleave(torch.FloatTensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1), N, dim=0)
    
    return torch.repeat_interleave(torch.FloatTensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1), N, dim=0)

def create_color_image(np_colors, l_hex_colors):
    """Create a colored image from index array and color palette.

    Args:
        np_colors (np.ndarray): 2D array of indices [H, W] where each value
            corresponds to a color in l_hex_colors. NaN values are filled with white.
        l_hex_colors (list[str]): List of hex color codes (e.g., "#FF0000").
            Can be generated from scanpy.pl.umap.

    Returns:
        np.ndarray: RGB image array of shape [H, W, 3] with uint8 values.
            Each pixel is colored according to the index in np_colors using
            the corresponding color from l_hex_colors.
    """
    rgb_colors = [mcolors.hex2color(hex_color) for hex_color in l_hex_colors]
    rgb_colors = np.array(rgb_colors) * 255
    rgb_colors = rgb_colors.astype(np.uint8)
    H, W = np_colors.shape

    image = 255+np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            if not np.isnan(np_colors[i, j]):
                idx = int(np_colors[i, j])
                image[i, j] = rgb_colors[idx]

    return image

def coords_to_mat(df_obs, l_hex_colors=None):
    """Convert coordinate-based observations to a matrix or colored image.

    Args:
        df_obs (pd.DataFrame): DataFrame containing 'x', 'y', and 'leiden' columns.
        l_hex_colors (list[str], optional): List of hex color codes. If provided,
            returns a colored image instead of index matrix. Defaults to None.

    Returns:
        np.ndarray: Either:
            - 2D array of leiden indices if l_hex_colors is None
            - RGB image array [H, W, 3] if l_hex_colors is provided
    """
    np_leiden_mat = pd.pivot_table(df_obs, index="x", columns="y")["leiden"].values
    if l_hex_colors is not None:
        np_leiden_mat = create_color_image(np_leiden_mat, l_hex_colors)
        
    return np_leiden_mat

