from typing import Dict, List, Tuple, Optional, Union

from tqdm import tqdm
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv
import trimesh
import networkx as nx
from scipy.spatial import cKDTree
from skimage import measure
from skimage import morphology

MAXINT = np.iinfo(np.int64).max

def subset_mesh(mesh: trimesh.Trimesh, sub_coords: Optional[List[int]]=None) -> trimesh.Trimesh:
    """Extract a subset of a mesh within specified coordinate bounds.

    Args:
        mesh (trimesh.Trimesh): Input mesh to subset.
        sub_coords (List[int], optional): Coordinate bounds as [x_min, x_max, y_min, y_max, z_min, z_max].
            If None, uses maximum possible bounds. Defaults to None.

    Returns:
        trimesh.Trimesh: New mesh containing only vertices and faces within the specified bounds.
            Returns empty mesh if no vertices fall within bounds.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    if sub_coords is None:
        sub_coords = [0, MAXINT, 0, MAXINT, 0, MAXINT]

    x_min, x_max, y_min, y_max, z_min, z_max = sub_coords
    if x_max < 0:
        x_max = MAXINT

    if y_max < 0:
        y_max = MAXINT

    if z_max < 0:
        z_max = MAXINT
        
    mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &\
            (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &\
            (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)

    valid_vertex_indices = np.where(mask)[0]
    valid_vertices_set = set(valid_vertex_indices)

    valid_faces = [face for face in faces if all(v in valid_vertices_set for v in face)]
    valid_faces = np.array(valid_faces)

    unique_vertices, inverse = np.unique(valid_faces, return_inverse=True)
    if len(unique_vertices) == 0:
        return trimesh.Trimesh(vertices=[], faces=[])
    
    new_vertices = vertices[unique_vertices]
    new_faces = inverse.reshape((-1, 3))
    subset_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    return subset_mesh


def filter_sub_meshes(
        mesh: trimesh.Trimesh, 
        min_vol: float=2000, 
        merge_threshold: float=10.0, 
        smooth: bool=True, 
        smooth_kwargs: Optional[dict]=None
    ) -> trimesh.Trimesh:
    """Filter and merge submeshes based on volume and proximity.

    Args:
        mesh (trimesh.Trimesh): Input mesh to process.
        min_vol (float, optional): Minimum volume threshold for keeping submeshes. 
            Defaults to 2000.
        merge_threshold (float, optional): Distance threshold for merging nearby submeshes.
            Defaults to 10.0.
        smooth (bool, optional): Whether to apply Laplacian smoothing to submeshes.
            Defaults to True.
        smooth_kwargs (dict, optional): Additional arguments for smoothing.
            Defaults to None.

    Returns:
        trimesh.Trimesh: Processed mesh with filtered and merged submeshes.
    """
    sub_meshes = mesh.split(only_watertight=False)
    l_filtered_mesh = []
    print(f"divide into {len(sub_meshes)} sub_meshes")
    for i, sub_mesh in enumerate(sub_meshes):
        volume = sub_mesh.volume
        vol = -volume
        if vol > min_vol:
            if smooth:
                if smooth_kwargs is None:
                    smooth_kwargs = {}

                sub_mesh = smooth_sub_mesh(sub_mesh, **smooth_kwargs)

            l_filtered_mesh.append(sub_mesh)

    print(f"filtered into {len(l_filtered_mesh)} sub_meshes")
    merged_sub_meshes = merge_submeshes_knn(
        l_filtered_mesh, threshold=merge_threshold, k=5, sample_points=min_vol)

    print(f"merged into {len(merged_sub_meshes)} sub_meshes")
    new_mesh = trimesh.util.concatenate(merged_sub_meshes)
    return new_mesh


def gaussian_3d_kernel(size: Tuple[int, int, int], sigma: float) -> torch.Tensor:
    """Generate a 3D Gaussian kernel.

    Args:
        size (Tuple[int, int, int]): Kernel dimensions (size_x, size_y, size_z).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Normalized 3D Gaussian kernel.
    """
    size_x, size_y, size_z = size
    x = torch.arange(size_x).float() - size_x // 2
    y = torch.arange(size_y).float() - size_y // 2
    z = torch.arange(size_z).float() - size_z // 2
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def apply_gaussian_3d_filter(
        input_tensor: torch.Tensor, 
        kernel_size: Union[int, Tuple[int, int, int]], 
        sigma: float
    ) -> torch.Tensor:
    """Apply 3D Gaussian filtering to a tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [N, C, D, H, W].
        kernel_size (Union[int, Tuple[int, int, int]]): Size of the Gaussian kernel.
            If int, uses same size in all dimensions.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        torch.Tensor: Filtered tensor of same shape as input.
    """
    channels = input_tensor.size(1)
    if type(kernel_size) is not tuple:
        kernel_size = (1, 1, kernel_size)

    kernel = gaussian_3d_kernel(kernel_size, sigma)
    kernel_size_x, kernel_size_y, kernel_size_z = kernel_size
    kernel = kernel.view(1, 1, kernel_size_z, kernel_size_x,
                         kernel_size_y).repeat(channels, 1, 1, 1, 1)
    padding = (kernel_size_x // 2, kernel_size_x // 2, kernel_size_y //
               2, kernel_size_y // 2, kernel_size_z // 2, kernel_size_z // 2)

    padded_input = F.pad(input_tensor, padding, mode='replicate')

    filtered_tensor = F.conv3d(padded_input, kernel, groups=channels)
    return filtered_tensor


def smooth_sub_mesh(
        sub_mesh: trimesh.Trimesh,
        n_iter: int=50,
        relaxation_factor: float=0.1,
        feature_angle: float=45.0
    ) -> trimesh.Trimesh:
    """Apply Laplacian smoothing to a mesh while preserving features.

    Args:
        sub_mesh (trimesh.Trimesh): Input mesh to smooth.
        n_iter (int, optional): Number of smoothing iterations. Defaults to 50.
        relaxation_factor (float, optional): Relaxation factor controlling smoothing strength.
            Defaults to 0.1.
        feature_angle (float, optional): Angle threshold for preserving sharp features.
            Defaults to 45.0.

    Returns:
        trimesh.Trimesh: Smoothed mesh.
    """
    pv_mesh = trimesh_to_pyvista(sub_mesh)
    smoothed_pv = pv_mesh.smooth(n_iter=n_iter,
                                 relaxation_factor=relaxation_factor,
                                 feature_angle=feature_angle,
                                 boundary_smoothing=True)
    return pyvista_to_trimesh(smoothed_pv)


def merge_submeshes_knn(
        sub_meshes: List[trimesh.Trimesh], 
        threshold: float=3.0, 
        k: int=5, 
        sample_points: int=1000
    ) -> List[trimesh.Trimesh]:
    """Merge nearby submeshes using k-nearest neighbors.

    This function merges submeshes that are within a specified distance threshold
    using the following strategy:
    1. Calculate centroids for each submesh
    2. Use KD-tree for efficient nearest neighbor search
    3. Build a graph where edges connect nearby meshes
    4. Merge meshes within each connected component

    Args:
        sub_meshes (List[trimesh.Trimesh]): List of meshes to merge.
        threshold (float, optional): Distance threshold for merging. Defaults to 3.0.
        k (int, optional): Number of nearest neighbors to check. Defaults to 5.
        sample_points (int, optional): Number of points to sample for distance calculation.
            Defaults to 1000.

    Returns:
        List[trimesh.Trimesh]: List of merged meshes.
    """
    n = len(sub_meshes)
    if n <= 1:
        return sub_meshes

    centroids = np.array([m.bounding_box.centroid for m in sub_meshes])
    tree = cKDTree(centroids)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in tqdm(range(n)):
        dist, idx = tree.query(centroids[i], k=k+1)
        for d, j in zip(dist[1:], idx[1:]):
            if j == i:
                continue

            if d > threshold:
                continue

            dist_ij = mesh_surface_distance(
                sub_meshes[i], sub_meshes[j], sample_points=sample_points)
            if dist_ij < threshold:
                G.add_edge(i, j)

    merged_meshes = []
    for component in nx.connected_components(G):
        sublist = [sub_meshes[idx] for idx in component]
        merged_mesh = trimesh.util.concatenate(sublist)
        merged_meshes.append(merged_mesh)

    return merged_meshes


def mesh_surface_distance(
        mesh_a: trimesh.Trimesh, 
        mesh_b: trimesh.Trimesh, 
        sample_points: int=1000
    ) -> float:
    """Calculate the minimum distance between two mesh surfaces.

    Args:
        mesh_a (trimesh.Trimesh): First mesh.
        mesh_b (trimesh.Trimesh): Second mesh.
        sample_points (int, optional): Number of points to sample from mesh_a.
            Defaults to 1000.

    Returns:
        float: Minimum distance between the two mesh surfaces.
    """
    points_a, _ = trimesh.sample.sample_surface(mesh_a, count=sample_points)
    closest, dist, _ = trimesh.proximity.closest_point(mesh_b, points_a)
    return dist.min()


def mesh_from_verts_faces(verts: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Create a trimesh from vertices and faces arrays.

    Args:
        verts (np.ndarray): Vertex coordinates of shape [N, 3].
        faces (np.ndarray): Face indices of shape [M, 3].

    Returns:
        trimesh.Trimesh: Created mesh with specified vertices and faces.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def volume_to_verts_faces(
        np_binary_vol: np.ndarray, 
        level: float=0.5, 
        min_size: int=1
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a binary volume to mesh vertices and faces using marching cubes.

    Args:
        np_binary_vol (np.ndarray): Binary volume array of shape [H, W, D].
        level (float, optional): Threshold level for isosurface. Defaults to 0.5.
        min_size (int, optional): Minimum size of objects to keep. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices: Array of vertex coordinates [N, 3]
            - faces: Array of face indices [M, 3]
    """
    filtered_volume = morphology.remove_small_objects(
        np_binary_vol, min_size=min_size)
    verts, faces, normals, values = measure.marching_cubes(
        filtered_volume,
        level=level
    )
    return verts, faces


def trimesh_to_pyvista(tri_mesh: trimesh.Trimesh) -> pv.PolyData:
    """Convert a trimesh mesh to PyVista format.

    Args:
        tri_mesh (trimesh.Trimesh): Input trimesh mesh.

    Returns:
        pv.PolyData: Converted PyVista mesh.
    """
    tri_faces = tri_mesh.faces
    n_faces = len(tri_faces)
    # [3, i1, i2, i3, 3, j1, j2, j3, ...]
    face_data = np.c_[np.full(n_faces, 3), tri_faces].ravel()

    pv_poly = pv.PolyData(tri_mesh.vertices, face_data)
    return pv_poly


def pyvista_to_trimesh(pv_poly: pv.PolyData) -> trimesh.Trimesh:
    """Convert a PyVista mesh to trimesh format.

    Args:
        pv_poly (pv.PolyData): Input PyVista mesh.

    Returns:
        trimesh.Trimesh: Converted trimesh mesh.
    """
    faces_array = pv_poly.faces.reshape((-1, 4))[:, 1:]
    points = pv_poly.points
    return trimesh.Trimesh(vertices=points, faces=faces_array, process=False)


def color_to_rgb(color_name: Union[str, float]) -> Union[List[int], float]:
    """Convert a color name or hex code to RGB values.

    Args:
        color_name (Union[str, float]): Color specification. Can be:
            - A color name (e.g., "red")
            - A hex code (e.g., "#FF0000")
            - A float value

    Returns:
        Union[List[int], float]: Either:
            - List of RGB values [0-255] if input is a color name/hex
            - Original float value if input is a float
    """
    if type(color_name) is float:
        return color_name

    if color_name.startswith('#'):
        return [int(color_name[i:i+2], 16) for i in (1, 3, 5)]

    color = mcolors.to_rgb(color_name)
    return [int(255 * c) for c in color]


def save_mesh_as_ply(
        mesh: Dict[str, np.ndarray], 
        color: Union[str, float], 
        alpha: float, 
        filename: str
    ):
    """Save a mesh to PLY format with vertex colors.

    Args:
        mesh (Dict[str, np.ndarray]): Dictionary containing:
            - 'verts': Vertex coordinates array
            - 'faces': Face indices array
        color (Union[str, float]): Color specification for vertices.
            Can be a color name, hex code, or float value.
        alpha (float): Alpha (opacity) value [0-1].
        filename (str): Output PLY file path.
    """
    trimesh_mesh = trimesh.Trimesh(
        vertices=mesh['verts'], faces=mesh['faces'], process=False)
    vertex_colors = np.tile(color_to_rgb(
        color) + [int(alpha * 255)], (trimesh_mesh.vertices.shape[0], 1))
    trimesh_mesh.visual.vertex_colors = vertex_colors
    trimesh_mesh.export(filename, file_type='ply')
