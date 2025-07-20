import logging as logging_
from typing import List

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from matplotlib import cm
import pyvista as pv

from omnialigner.utils.mesh import apply_gaussian_3d_filter, volume_to_verts_faces, mesh_from_verts_faces, filter_sub_meshes, color_to_rgb
from omnialigner.dtypes import Np_image_HWC

pv.global_theme.trame.server_proxy_enabled = True
pv.set_jupyter_backend('client')


logging_.getLogger('trame_server').propagate = False
logging_.getLogger('trame_client').propagate = False

def plot_nhwc_3d(np_nhwc):
    """Plot 3D volume data in NHWC format using PyVista.

    Args:
        np_nhwc: Input numpy array in NHWC format (height, width, depth, channels).

    Returns:
        pyvista.Plotter: PyVista plotter object with the rendered volume.
    """
    plotter = pv.Plotter(notebook=True, window_size=(800, 600))

    ## generate pseudo volume data

    h, w, z, c = np_nhwc.shape
    points = np.arange(h*w*z).reshape(h, w, z, order='F')
    colors = np_nhwc.reshape(-1, c)
    
    volume = pv.helpers.wrap(points)
    volume["Data"] = colors
    volume.active_scalars_name = "Data"

    plotter.add_mesh_clip_plane(volume, rgb=True)
    plotter.view_xy()
    plotter.show()


def crop_cubic(img, crop=(0, -1, 0, -1, 0, -1)):
    img[0:crop[0], :, 3] = 0
    img[crop[1]:, :, 3] = 0
    img[:, 0:crop[2], 3] = 0
    img[:, crop[3]:, 3] = 0
    return img

def stack_rgba_images(
        l_rgba_images,
        np_z_pos=None,
        l_kpts=None,
        down_scale=5,
        window_size=(800, 600, 200),
        crop=None,
        kpt_kwargs=None
    ):
    """Stack and visualize multiple RGBA images in 3D.

    Args:
        l_rgba_images: List of RGBA images to stack.
        np_z_pos: Optional array of z-positions for each image.
        l_kpts: Optional list of keypoints for each image.
        down_scale: Downscaling factor (default: 5).
        window_size: Window size as (width, height, depth) (default: (800, 600, 200)).
        crop: Optional cropping parameters.
        kpt_kwargs: Optional keypoint visualization parameters.

    Returns:
        pyvista.Plotter: PyVista plotter object with the stacked images.
    """
    kpt_kwargs = {"color": "k", "line_width": 2} if kpt_kwargs is None else kpt_kwargs
    plotter = pv.Plotter(notebook=True, window_size=window_size[0:2])
    plotter.add_axes()
    l_color_default = ["blue", "red"]
    plotter.view_isometric()
    n = len(l_rgba_images)
    for idx, img in tqdm(enumerate(l_rgba_images)):
        img = img[::-1, :, :]
        if crop is not None:
            img = crop_cubic(img, crop=crop)

        w, h, c = img.shape
        z_interval = window_size[2] / (n+1) * idx
        if np_z_pos is not None:
            z_interval = np_z_pos[idx]

        img_plane = pv.Plane(center=(h//(2*down_scale), w//(2*down_scale), z_interval),
                             direction=(0, 0, 1),
                             i_size=(h//down_scale),
                             j_size=(w//down_scale))

        # texture = pv.numpy_to_texture(np.moveaxis(img, 0, 1))
        texture = pv.numpy_to_texture(img)
        plotter.add_mesh(img_plane, texture=texture)
        if l_kpts and l_kpts[idx] is not None:
            for kpt in l_kpts[idx]:
                x, y = kpt / down_scale
                z = z_interval * idx
                plotter.add_mesh(pv.Sphere(center=(x, y, z), radius=5), color=l_color_default[idx % 2])

    if l_kpts and len(l_kpts) > 1 and (kpt_kwargs is not None):
        color = kpt_kwargs.get("color", "black")
        line_width = kpt_kwargs.get("line_width", 2)
        kpts1 = l_kpts[0] / down_scale
        kpts2 = l_kpts[1] / down_scale
        z1 = z_interval * 0
        z2 = z_interval * 1
        for kpt1, kpt2 in zip(kpts1, kpts2):
            x1, y1 = kpt1
            x2, y2 = kpt2
            line = pv.Line((x1, y1, z1), (x2, y2, z2))
            plotter.add_mesh(line, color=color, line_width=line_width)

        
    plotter.view_xy()
    plotter.show()


def matrix_to_multi_png(matrix: np.ndarray, cmap: List[str] = None):
    """
    Converts a matrix to multiple PNG layers represented as an array.

    Args:
        matrix (np.ndarray): Input matrix of shape (H, W).
        cmap: List of colors to use for different values.

    Returns:
        np.ndarray: Array of shape (N, H, W, 3), where N is the number of unique values in the matrix excluding zero, and C=3 for RGB.
    """
    unique_values = range(1, np.max(matrix) + 1)
    h, w = matrix.shape
    l_layers = []
    for value in unique_values:
        layer = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask = (matrix == value)
        if np.any(mask):
            color = get_color_from_cmap(cmap, value)
            layer[mask] = color
        l_layers.append(layer)
    return np.stack(l_layers)


def get_color_from_cmap(cmap: List[str], value: int) -> np.ndarray:
    """
    Retrieves a color from the colormap, cycling through the list if the index exceeds the length.

    Args:
        cmap (List[str]): List of color hex strings.
        value (int): Index to retrieve the color for.

    Returns:
        np.ndarray: RGB color as a numpy array.
    """
    index = value % len(cmap)
    print(cmap[index])
    return np.array(color_to_rgb(cmap[index]))
    # return np.array([int(cmap[index][i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def add_border_to_images(images: np.ndarray, border_thickness: int = 10, border_color: np.ndarray = None) -> np.ndarray:
    """
    Adds a border to each image in the given array.

    Args:
        images (np.ndarray): Input images of shape (N, H, W, C).
        border_thickness (int): Thickness of the border.
        border_color (np.ndarray): Color of the border in RGB format.

    Returns:
        np.ndarray: Images with added border of shape (N, H + 2 * border_thickness, W + 2 * border_thickness, C).
    """

    N, H, W, C = images.shape
    if border_color is None:
        border_color = np.zeros([C])

    new_H = H + 2 * border_thickness
    new_W = W + 2 * border_thickness
    padded_images = np.ones((N, new_H, new_W, C), dtype=np.uint8) * 255
    for i in range(N):
        padded_images[i, :border_thickness, :, :] = border_color
        padded_images[i, -border_thickness:, :, :] = border_color
        padded_images[i, :, :border_thickness, :] = border_color
        padded_images[i, :, -border_thickness:, :] = border_color
        padded_images[i, border_thickness:border_thickness+H,
                      border_thickness:border_thickness+W, :] = images[i]
    return padded_images


def plot_uint8_matrix(matrix: np.ndarray, pv=None, layer_spacing: int = 5, border_thickness: int = 1, cmap: List[str] = None):
    """
    Plots a uint8 matrix using PyVista, with each unique value represented as a separate layer.

    Args:
        matrix (np.ndarray): Input matrix of shape (H, W) contains label in position.
        layer_spacing (int): Spacing between layers.
        border_thickness (int): Thickness of the border around each layer.
        cmap: List of colors to use for different values.
    """
    if cmap is None:
        cmap = [cm.colors.rgb2hex(c) for c in cm.get_cmap(
            'Set1', 21)(np.linspace(0, 1, 21))]

    images = matrix_to_multi_png(matrix, cmap=cmap)
    border_color = np.array([0, 0, 0], dtype=np.uint8)
    images = add_border_to_images(images, border_thickness, border_color)
    plotter = pv.Plotter(notebook=True)
    N, H, W, C = images.shape
    for idx, image in enumerate(images):
        grid = pv.ImageData(
            dimensions=(W, H, 1),
            spacing=(1, 1, 1),
            origin=(0, 0, (N - idx - 1) * layer_spacing)
        )
        scalars = image.reshape(-1, 3)
        grid.point_data['colors'] = scalars
        plotter.add_mesh(grid, scalars='colors', rgb=True,
                         opacity=1.0, show_edges=False, lighting=False)
    plotter.enable_parallel_projection()
    plotter.set_background("white")
    plotter.show()


class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, state):
        self.actor.SetVisibility(state)


def show_pv_multiple_volumes(volume_meshes: List, img: np.ndarray = None, scale=0.2):
    """
    Args:
    - volume_meshes: list of dict
        {
          "verts": 顶点数组,
          "faces": 面索引数组,
          "color": "red" or (R, G, B),
          "alpha": float (0~1)
        }
    - img:  bottom HWC uint8 image

    """
    # plotter = pv.Plotter(notebook=True, window_size=(800, 600))
    plotter = pv.Plotter(window_size=(800, 600))

    start_pos = 12
    widget_size = 50
    for idx, mesh_dict in enumerate(volume_meshes):
        verts = mesh_dict["verts"]
        faces = mesh_dict["faces"]
        color = mesh_dict.get("color", "white")
        alpha = mesh_dict.get("alpha", 1.0)

        faces_pv = np.hstack([
            np.full((faces.shape[0], 1), 3),
            faces
        ]).astype(np.int64).flatten()

        mesh = pv.PolyData(verts, faces_pv)
        # save_mesh_as_ply(mesh_dict, color, alpha, f"{idx}.mesh.ply")
        actor = plotter.add_mesh(
            mesh,
            color=color,
            opacity=alpha,
            show_edges=False
        )
        callback = SetVisibilityCallback(actor)
        plotter.add_checkbox_button_widget(
            callback,
            value=True,
            position=(5.0, start_pos),
            size=widget_size,
            border_size=1,
            color_on=color,
            color_off='grey',
            background_color='grey',
        )
        start_pos = start_pos + widget_size + (widget_size // 10)

    plotter.add_axes()

    if img is not None:
        h, w, c = img.shape
        img_plane = pv.Plane(center=((h*scale)//2, (w*scale)//2, 2.5),
                             direction=(0, 0, 1),
                             i_size=int(h*scale),
                             j_size=int(w*scale))

        texture = pv.numpy_to_texture(img[::-1, :, :])
        plotter.add_mesh(img_plane, texture=texture)

    plotter.view_isometric()
    plotter.add_bounding_box()
    plotter.show()


def search_inner_hole(image, iterations=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_ret1 = cv2.dilate(image, kernel, iterations=iterations)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    image = cv2.erode(img_ret1, kernel, iterations=iterations)
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    result = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(result, contours[1:], -1, 255, thickness=cv2.FILLED)
    return result


def bin_to_mesh(
        np_binary_vol: Np_image_HWC,
        coord_z: np.array,
        min_size: int=10,
        level: float=0.1,
        gaussian_kernel_size: int=5,
        guassian_sigma: int=15,
        binary_cutoff: float=0.5,
        merge_threshold: float=10.0,
        smooth: bool=True,
        smooth_kwargs: dict=None,
        zmax: int=750,
        z_scale: float=1.0,
        down_scale: int=5
    ):
    """Convert binary volume to 3D mesh representation.

    Args:
        np_binary_vol: Binary volume data.
        coord_z: Z-coordinates for the volume.
        min_size: Minimum size for mesh components (default: 10).
        level: Isosurface level (default: 0.1).
        gaussian_kernel_size: Size of Gaussian smoothing kernel (default: 5).
        guassian_sigma: Sigma for Gaussian smoothing (default: 15).
        binary_cutoff: Threshold for binary conversion (default: 0.5).
        merge_threshold: Threshold for merging mesh components (default: 10.0).
        smooth: Whether to smooth the mesh (default: True).
        smooth_kwargs: Optional smoothing parameters.
        zmax: Maximum z-coordinate (default: 750).
        z_scale: Z-axis scaling factor (default: 1.0).
        down_scale: Downscaling factor (default: 5).

    Returns:
        pyvista.PolyData: The generated 3D mesh.
    """

    z_scale = 1/(40*down_scale) * (4/0.25)
    data = torch.from_numpy(np_binary_vol).float(
    ).unsqueeze(0).permute(0, 3, 1, 2)
    np_binary_vol = F.interpolate(
        data, [1280//down_scale, 1280//down_scale]).permute(0, 2, 3, 1).squeeze(0).numpy()

    input_tensor = torch.from_numpy(np_binary_vol).unsqueeze(
        0).unsqueeze(0).permute(0, 1, 4, 2, 3)
    smoothed_tensor = apply_gaussian_3d_filter(
        input_tensor, gaussian_kernel_size, sigma=guassian_sigma)

    voxel_data = smoothed_tensor[0, 0, :, :, :].permute(
        1, 2, 0).numpy() > binary_cutoff

    h, w, n = np_binary_vol.shape
    idx = np.array(coord_z) < zmax

    verts, faces = volume_to_verts_faces(
        voxel_data[:, :, idx], level=level, min_size=0.1*min_size)
    z = np.array([coord_z[int(x)] for x in verts[:, 2]])
    verts[:, 2] = z_scale*(z-np.min(z))

    mesh = mesh_from_verts_faces(verts, faces)
    mesh = filter_sub_meshes(mesh, min_vol=min_size, smooth=smooth,
                             smooth_kwargs=smooth_kwargs, merge_threshold=merge_threshold)
    return mesh


if __name__ == "__main__":
    uint8_matrix = np.random.randint(0, 21, size=(100, 100), dtype=np.uint8)
    plot_uint8_matrix(uint8_matrix, layer_spacing=10, border_thickness=1)
