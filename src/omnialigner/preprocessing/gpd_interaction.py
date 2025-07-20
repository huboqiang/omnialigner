from typing import List, Tuple
from collections import defaultdict

import cv2
import scanpy as sc
import pandas as pd
import numpy as np
import igraph as ig
from tqdm import tqdm
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union, transform
from shapely.affinity import scale
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors

from omnialigner.plotting.h5ad_viz import adata_to_gpd, gdf_shape_to_image


def binary_mask_to_geopandas(image_mask, scale_factor=10, min_area=5000, kernel_size: Tuple[int, int]=None, merge_polygons=False) -> gpd.GeoDataFrame:
    """
    Convert a binary mask to a GeoPandas GeoDataFrame.

    Args:
        image_mask: Binary mask image.
        scale_factor: Scaling factor, since img_obj is downsampled.
        min_area: Minimum area threshold to filter small contours.
        kernel_size: Size of the kernel for morphological operations (default is (3, 3)).
        merge_polygons: Whether to merge all polygons into a single geometry.

    Returns:
        gdf: GeoPandas GeoDataFrame.
    """
    if kernel_size is None:
        kernel_size = (3, 3)
    
    kernel = np.ones(kernel_size, np.uint8)
    mask_eroded = cv2.dilate(image_mask.astype(np.uint8), kernel, iterations=1)
    mask_dilated = cv2.erode(mask_eroded, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    valid_polygons = []
    for contour in contours:
        points = contour.squeeze() * scale_factor
        if len(points) < 3:
            continue

        try:
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    continue

            if polygon.area < min_area:
                continue

            valid_polygons.append(polygon)

        except Exception as e:
            continue
    
    if not valid_polygons:
        return gpd.GeoDataFrame({'contour_id': [], 'area': [], 'geometry': []})
    
    if merge_polygons:
        # 合并所有多边形为一个几何体
        if len(valid_polygons) == 1:
            merged_geom = valid_polygons[0]
        else:
            merged_geom = unary_union(valid_polygons)
        
        gdf = gpd.GeoDataFrame({
            'contour_id': [0],
            'area': [merged_geom.area],
            'n_components': [len(valid_polygons)],
            'geometry': [merged_geom]
        })
    else:
        gdf = gpd.GeoDataFrame({
            'contour_id': range(len(valid_polygons)),
            'area': [p.area for p in valid_polygons],
            'geometry': valid_polygons
        })
    
    return gdf



def adata_to_regions(adata_sc: sc.AnnData, key: str="CAST_label", radius: int=16, scale_factor: int = 10, min_area: int = 5000, image_size: Tuple[int, int]=None) -> gpd.GeoDataFrame:
    """
    Convert an AnnData object to a GeoPandas GeoDataFrame with regions.

    Args:
        adata_sc: Annotated data object with spatial coordinates in adata_sc.obsm["spatial"]
        key: Key for the label in adata.obs
        scale_factor: scaling factor for downsampling the image
        min_area: Minimum area threshold to filter small contours
        image_size: Size of the image to create the mask, if None, it will be determined from the coordinates.
    Returns:
        gdf: GeoPandas GeoDataFrame
    """
    gdf_obj = adata_to_gpd(adata_sc, key=key, radius=radius)
    key_new = f"{key}_num"
    if image_size is None:
        coords = adata_sc.obsm["spatial"]
        image_size = coords.max(1).astype(int)
    
    if not pd.api.types.is_integer_dtype(gdf_obj[key]):
        gdf_obj[key] = gdf_obj[key].astype("category")
        gdf_obj[key_new] = gdf_obj[key].cat.codes
    else:
        gdf_obj[key_new] = gdf_obj[key].astype(int)

    gdf_obj[key_new] = gdf_obj[key_new] + 1
    img_obj = gdf_shape_to_image(gdf_obj, key=key_new, h=image_size[0], w=image_size[1])
    img_obj = cv2.resize(img_obj.astype(np.uint8), (img_obj.shape[1]//scale_factor, img_obj.shape[0]//scale_factor))
    
    l_gdfs = []
    for i_layer in range(min(gdf_obj[key_new]), max(gdf_obj[key_new])+1):
        ## img_mask: is one cell type or is gene-high exp
        img_mask = (img_obj == i_layer).astype(np.uint8)
        gdf_contours = binary_mask_to_geopandas(img_mask, scale_factor=scale_factor, min_area=min_area, merge_polygons=False)
        gdf_contours[key_new] = i_layer - 1
        l_gdfs.append(gdf_contours)

    gdf_all = pd.concat(l_gdfs, ignore_index=True)
    gdf_all['geometry'] = gdf_all['geometry'].apply(lambda geom: shift_geometry(geom, radius, radius))

    return gdf_all

def calculate_region_signal(
        gdf_obj: gpd.GeoDataFrame,
        img_exp: np.ndarray,
        l_genes: List[str],
        scale_factor: float = 10) -> pd.DataFrame:
    """
    Calculate the average signal of each gene in the region defined by gdf_obj.

    Args:
        gdf_obj: GeoDataFrame containing geometries of regions.
        img_exp: 2D numpy array representing the image data (e.g., expression values).
        l_genes: List of gene names corresponding to the channels in img_exp.
        scale_factor: Scaling factor for downsampling the image.
    
    Returns:
        DataFrame with average signal for each region and gene.
    """
    gdf_obj = gdf_obj.copy()
    gdf_obj["clusters"] = 1
    l_mat = []

    for i in tqdm(range(len(gdf_obj))):
        geom = gdf_obj.iloc[i].geometry

        if not geom.is_valid:
            geom = geom.buffer(0)
            if not geom.is_valid:
                print(f"Warning: Original geometry {i} is invalid and cannot be fixed.")
                continue

        if geom.is_empty:
            print(f"Warning: Original geometry {i} is empty.")
            continue


        if isinstance(geom, MultiPolygon):
    
            try:
                geom = unary_union(geom)
                if isinstance(geom, MultiPolygon):
            
                    geom = max(geom.geoms, key=lambda p: p.area)
            except Exception as e:
                print(f"Warning: Failed to process MultiPolygon {i}: {e}")
                continue
        

        try:
            scaled_geom = scale(geom, xfact=1/scale_factor, yfact=1/scale_factor, origin=(0, 0))
        except Exception as e:
            print(f"Warning: Failed to scale geometry {i}: {e}")
            continue
        

        if not scaled_geom.is_valid:
            scaled_geom = scaled_geom.buffer(0)
            if not scaled_geom.is_valid:
                print(f"Warning: Scaled geometry {i} is invalid and cannot be fixed.")
                continue

        if scaled_geom.is_empty:
            print(f"Warning: Scaled geometry {i} is empty.")
            continue

        if isinstance(scaled_geom, MultiPolygon):
            try:
                scaled_geom = unary_union(scaled_geom)
                if isinstance(scaled_geom, MultiPolygon):
                    scaled_geom = max(scaled_geom.geoms, key=lambda p: p.area)
            except Exception as e:
                print(f"Warning: Failed to process scaled MultiPolygon {i}: {e}")
                continue
        
        bounds = scaled_geom.bounds
        if (bounds[2] <= 0 or bounds[3] <= 0 or 
            bounds[0] >= img_exp.shape[1] or bounds[1] >= img_exp.shape[0]):
            print(f"Warning: Geometry {i} is outside image bounds.")
            continue
        
        if scaled_geom.area < 1.0:
            print(f"Warning: Geometry {i} area too small after scaling: {scaled_geom.area}")
            continue

        gdf_temp = gpd.GeoDataFrame({'clusters': [1]}, geometry=[scaled_geom])
        try:
            mask_region = gdf_shape_to_image(
                gdf_temp, 
                key="clusters", 
                h=img_exp.shape[0], 
                w=img_exp.shape[1]
            )[:, :, np.newaxis]
            
    
            if mask_region.sum() == 0:
                print(f"Warning: Geometry {i} produced empty mask.")
                continue
                
        except Exception as e:
            print(f"Warning: Failed to rasterize geometry {i}: {e}")
            print(f"  Geometry type: {type(scaled_geom)}")
            print(f"  Geometry bounds: {scaled_geom.bounds}")
            print(f"  Geometry area: {scaled_geom.area}")
            continue

        try:
            masked_exp = np.where(mask_region > 0, img_exp, np.nan)
            np_avg = np.nanmean(masked_exp, axis=(0, 1))

            if np.isnan(np_avg).all():
                print(f"Warning: All NaN values for geometry {i}")
                continue

            l_mat.append(np_avg)

        except Exception as e:
            print(f"Warning: Failed to compute signal for geometry {i}: {e}")
            continue

    if not l_mat:
        print("Warning: No valid geometries were processed.")
        return pd.DataFrame(columns=l_genes + ["area"])

    df_gdf_exp = pd.DataFrame(l_mat, columns=l_genes)
    if len(l_mat) == len(gdf_obj):
        df_gdf_exp["area"] = gdf_obj.area.values
    else:
        print(f"Warning: Only {len(l_mat)}/{len(gdf_obj)} geometries were successfully processed.")
        df_gdf_exp["area"] = np.nan

    return df_gdf_exp


def shift_geometry(geom, dx=25, dy=25):
    def shift_coords(x, y, z=None):
        return x + dx, y + dy
    
    return transform(shift_coords, geom)

def create_outline(
        gdf: gpd.GeoDataFrame,
        thickness: float=50,
        simplify_tolerance: float=None,
        keep_attributes: bool=True
    ):
    """
    Create an outline of specified thickness for each geometry in a GeoDataFrame.
    if `thickness` is positive, it creates an outer outline; if negative, it creates an inner outline.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.
        thickness (float): Outline thickness in pixels. Positive for outer, negative for inner outline.
        simplify_tolerance (float, optional): Tolerance for geometry simplification.
        keep_attributes (bool): Whether to keep original attribute columns.

    Returns:
        GeoDataFrame: GeoDataFrame with outline geometries, same number of rows as input.
    """
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    outlines = []

    for idx, row in gdf.iterrows():
        geom = row.geometry

        # Optionally simplify geometry
        if simplify_tolerance is not None and hasattr(geom, 'simplify'):
            geom = geom.simplify(simplify_tolerance)

        # Create buffer and outline
        if thickness >= 0:
            outer_buffer = geom.buffer(thickness)
            inner_buffer = geom.buffer(0)
            outline = outer_buffer.difference(inner_buffer)
        else:
            outer_buffer = geom.buffer(0)
            inner_buffer = geom.buffer(thickness)
            outline = outer_buffer.difference(inner_buffer)

        outline_data = {'geometry': outline}

        if keep_attributes:
            for col in gdf.columns:
                if col != 'geometry':
                    outline_data[col] = row[col]

        outlines.append(outline_data)

    outline_gdf = gpd.GeoDataFrame(outlines, crs=gdf.crs)    
    return outline_gdf


def get_cells_in_outline(adata_sc, outline_gdf, predicate='within', tag_roi="contour_id"):
    """
    Extract cells from an AnnData object that are within or intersecting a given outline.
    
    Args:
        adata_sc: AnnData object containing spatial coordinates in adata_sc.obsm["spatial"]
        outline_gdf: GeoDataFrame containing outline geometries
        predicate: Spatial predicate, can be 'within' or 'intersects'
    
    Returns:
        subset_adata: AnnData object containing only cells within the outline
        cell_indices: Indices of cells within the outline
    """
    # Create a GeoDataFrame of cell points
    coords = adata_sc.obsm["spatial"]
    points_gdf = gpd.GeoDataFrame(
        {'cell_id': range(len(coords))},
        geometry=[Point(x, y) for x, y in coords]
    )
    
    # Use spatial join to find points within the outline
    joined = gpd.sjoin(points_gdf, outline_gdf, how='inner', predicate=predicate)
    dict_joined = { k:v for k,v in zip(joined["cell_id"],joined[tag_roi]) }
    cell_indices = joined['cell_id'].unique()

    # Return the corresponding AnnData subset
    subset_adata = adata_sc[cell_indices].copy()
    return subset_adata, cell_indices, dict_joined

def find_knn_between_groups(
        coords_A: np.ndarray,
        coords_B: np.ndarray,
        k: int=5,
        distance_threshold: float=None
    ):
    """
    Quickly find k-nearest neighbor relationships between two groups of cells.

    Args:
        coords_A: Coordinates of the first group of cells [N_A, 2]
        coords_B: Coordinates of the second group of cells [N_B, 2]
        k: Number of nearest neighbors in group B for each cell in group A
        distance_threshold: Distance threshold; connections exceeding this distance will be filtered out

    Returns:
        edges: List of tuples (idx_A, idx_B, distance)
    """
    if len(coords_A) == 0 or len(coords_B) == 0:
        return []

    # Build KD-tree for group B
    nbrs = NearestNeighbors(n_neighbors=min(k, len(coords_B)), algorithm='kd_tree')
    nbrs.fit(coords_B)

    # Find nearest neighbors in group B for each point in group A
    distances, indices = nbrs.kneighbors(coords_A)

    edges = []
    for i_A in range(len(coords_A)):
        for j, (dist, i_B) in enumerate(zip(distances[i_A], indices[i_A])):
            if distance_threshold is None or dist <= distance_threshold:
                edges.append((i_A, i_B, dist))

    return edges


def tissue_edge_cell_interaction(
        adata_sc: sc.AnnData,
        gdf_tissue: gpd.GeoDataFrame,
        key: str = "cell_type",
        tag_roi: str = "contour_id",
        k_neighbors: int = 10,
        cell_distance_threshold: int = 50,
        thickness: float=50
    ) -> pd.DataFrame:
    outline_gdf = create_outline(gdf_tissue, thickness=thickness)
    inline_gdf = create_outline(gdf_tissue, thickness=-thickness)
    subset_adata_outer, cell_indices_outer, dict_cellroi_out = get_cells_in_outline(adata_sc, outline_gdf, tag_roi=tag_roi)
    subset_adata_inner, cell_indices_inner, _ = get_cells_in_outline(adata_sc, inline_gdf, tag_roi=tag_roi)

    coords_outer = subset_adata_outer.obsm["spatial"]
    coords_inner = subset_adata_inner.obsm["spatial"]
    edges_i_to_j = find_knn_between_groups(coords_outer, coords_inner, k_neighbors, cell_distance_threshold)
    all_cell_edges = []
    for i_local, j_local, dist in edges_i_to_j:
        cell_i = cell_indices_outer[i_local]
        cell_j = cell_indices_inner[j_local]
        l_cmp = sorted([adata_sc.obs.iloc[cell_i][key], adata_sc.obs.iloc[cell_j][key]])
        all_cell_edges.append({
            'cell_i': cell_i,
            'cell_j': cell_j,
            'distance': dist,
            f'{key}_i': adata_sc.obs.iloc[cell_i][key],
            f'{key}_j': adata_sc.obs.iloc[cell_j][key],
            f'roi_i': dict_cellroi_out.get(cell_i, "NA"),
            'cell_interaction_type': f"{l_cmp[0]}-{l_cmp[1]}"
        })

    cell_interactions_df = pd.DataFrame(all_cell_edges)
    return cell_interactions_df

def interaction_to_igraph(
        cell_interactions_df: pd.DataFrame,
        key: str = "cell_type",
        l_exceptions: List[str] = None
    ) -> ig.Graph:
    all_tissues = list(
        set(cell_interactions_df[f'{key}_i'].tolist() + cell_interactions_df[f'{key}_j'].tolist())
    )
    if l_exceptions is not None:
        all_tissues = [tissue for tissue in all_tissues if tissue not in l_exceptions]
        cell_interactions_df = cell_interactions_df[
            ~cell_interactions_df[f'{key}_i'].isin(l_exceptions) & 
            ~cell_interactions_df[f'{key}_j'].isin(l_exceptions)
        ]

    # Create adjacency matrix for tissue interactions
    tissue_interactions = defaultdict(int)
    for _, row in cell_interactions_df.iterrows():
        pair = tuple(sorted([row[f'{key}_i'], row[f'{key}_j']]))
        tissue_interactions[pair] += 1

    # Create edge list
    edges = []
    edge_weights = []
    for (tissue1, tissue2), count in tissue_interactions.items():
        edges.append((tissue1, tissue2))
        edge_weights.append(count)

    # Create igraph
    g = ig.Graph()
    g.add_vertices(all_tissues)
    g.add_edges(edges)
    g.es['weight'] = edge_weights

    # Set node attributes
    g.vs['label'] = all_tissues
    g.vs['size'] = [cell_interactions_df[cell_interactions_df[f'{key}_i'] == tissue].shape[0] +
                    cell_interactions_df[cell_interactions_df[f'{key}_j'] == tissue].shape[0]
                    for tissue in all_tissues]

    # Set edge width
    g.es['width'] = [w * 2 for w in edge_weights]
    return g
