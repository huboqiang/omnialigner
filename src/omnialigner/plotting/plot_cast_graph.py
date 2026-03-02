from typing import Dict, List, Tuple, Set
from collections import deque, defaultdict

from scipy import sparse
import cv2
import networkx as nx
import scanpy as sc
import pandas as pd
import numpy as np
from tqdm import tqdm
import igraph as ig
from shapely.geometry import Point, box, Polygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
import geopandas as gpd
from multiprocessing import Pool, cpu_count

import omnialigner as om
from omnialigner.plotting.h5ad_viz import adata_to_gpd, gdf_shape_to_image


plt = om.pl.plt



def shift_geometry(geom, dx=25, dy=25):
    from shapely.ops import transform
    
    def shift_coords(x, y, z=None):
        return x + dx, y + dy
    
    return transform(shift_coords, geom)


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


def calculate_region_signal(gdf_obj: gpd.GeoDataFrame, img_exp: np.ndarray, l_genes: List[str], scale_factor: float = 10):
    """
    Calculate the average signal of each gene in the region defined by gdf_obj.
    """
    from shapely.affinity import scale
    from shapely.geometry import MultiPolygon, Polygon
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

def adata_to_regions(adata_sc: sc.AnnData, key: str="CAST_label", radius: int=16, scale_factor: int = 10, min_area: int = 5000, kernel_size: Tuple[int, int]=None, image_size: Tuple[int, int]=None, basis: str="spatial") -> gpd.GeoDataFrame:
    """
    Convert an AnnData object to a GeoPandas GeoDataFrame with regions.

    Args:
        adata_sc: Annotated data object with spatial coordinates in adata_sc.obsm["spatial"]
        key: Key for the label in adata.obs
        scale_factor: scaling factor for downsampling the image
        min_area: Minimum area threshold to filter small contours
        kernel_size: Size of the kernel for morphological operations (default is (3, 3))
        image_size: Size of the output image (height, width). If None, use max coordinates from adata_sc.obsm["spatial"]
    Returns:
        gdf: GeoPandas GeoDataFrame
    """
    gdf_obj = adata_to_gpd(adata_sc, key=key, radius=radius)
    key_new = f"{key}_num"
    if kernel_size is None:
        kernel_size = (3, 3)

    if image_size is None:
        coords = adata_sc.obsm[basis]
        image_size = coords.max(0).astype(int)
    
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
        #
        img_mask = (img_obj == i_layer).astype(np.uint8)
        gdf_contours = binary_mask_to_geopandas(img_mask, scale_factor=scale_factor, min_area=min_area, kernel_size=kernel_size, merge_polygons=False)
        gdf_contours[key_new] = i_layer - 1
        l_gdfs.append(gdf_contours)

    gdf_all = pd.concat(l_gdfs, ignore_index=True)
    gdf_all['geometry'] = gdf_all['geometry'].apply(lambda geom: shift_geometry(geom, radius, radius))

    return gdf_all


def add_nodes_to_regions(
    gdf_regions: gpd.GeoDataFrame, 
    adata_sc: sc.AnnData,
    nodes_col: str = "nodes"
) -> gpd.GeoDataFrame:
    from shapely.geometry import Point
    import geopandas as gpd
    
    coords = adata_sc.obsm["spatial"]
    points_gdf = gpd.GeoDataFrame(
        {'cell_id': range(len(coords))},
        geometry=[Point(x, y) for x, y in coords]
    )
    
    joined = gpd.sjoin(points_gdf, gdf_regions, how='inner', predicate='within')
    
    nodes_dict = joined.groupby('index_right')['cell_id'].apply(list).to_dict()
    
    nodes_list = []
    for idx in gdf_regions.index:
        nodes_list.append(nodes_dict.get(idx, []))
    
    gdf_regions[nodes_col] = nodes_list
    
    return gdf_regions


def add_obs_labels_to_graph(
    G: nx.Graph,
    adata: sc.AnnData,
    obs_col: str      = "tissue", 
    label_attr: str   = "tissue", 
):
    """
    Set node attributes in a graph G based on adata.obs values.
    """

    all_int = all(isinstance(n, int) for n in G.nodes())
    max_id  = max(G.nodes()) if all_int else -1
    same_len= (max_id + 1 == len(G)) if all_int else False

    if all_int and same_len:

        for n in G.nodes():
            G.nodes[n][label_attr] = adata.obs[obs_col].iat[n]
    else:

        if len(G) != adata.n_obs:
            raise ValueError("节点数量与 adata.obs 行数不同，且无法用 ID 对应")
        for n, idx in zip(G.nodes(), range(adata.n_obs)):
            G.nodes[n][label_attr] = adata.obs[obs_col].iat[idx]


def region_anno(adata_sc, all_regions, label="tissue"):
    l_labels = []
    for i_region in range(all_regions.shape[0]):
        idx = np.array(all_regions.loc[i_region]["nodes"])
        value_counts = adata_sc[idx, :].obs[label].value_counts()
        if len(value_counts) == 0:
            label_name = "NA"
        elif len(value_counts) == 1:
            label_name = value_counts.index[0]
        else:
            if value_counts.index[0] == "Undefined" and len(value_counts) > 1:
                label_name = value_counts.index[1]
            else:
                label_name = value_counts.index[0]
        
        l_labels.append(label_name)

    all_regions[f"label_{label}"] = l_labels

def find_enclosed_regions(
    G: nx.Graph,
    tissue_attr: str = "CAST_label",
    inner_thr: float = 0.5
) -> pd.DataFrame:
    """
    Detect enclosed tissue regions in a spatial network graph.

    Args:
        G: Spatial network graph
        tissue_attr: Node attribute name for tissue type
        inner_thr: Maximum allowed proportion of neighboring nodes with different tissue type during region growing (0-1)

    Returns:
        all_regions: DataFrame with statistics for all regions,
    """
    labels = nx.get_node_attributes(G, tissue_attr)
    diff_ratio = {}
    for node in G.nodes:
        nbrs = list(G.neighbors(node))
        if not nbrs:
            diff_ratio[node] = 1.0
            continue
        diff_count = sum(labels[nbr] != labels[node] for nbr in nbrs)
        diff_ratio[node] = diff_count / len(nbrs)

    visited = set()
    regions_info = []
    contours_dict = {}
    surround_counts = {}
    surround_nodes_dict = {}
    region_id = 0

    for tissue_type in set(labels.values()):
        tissue_nodes = [n for n, t in labels.items() if t == tissue_type]
        for seed in tissue_nodes:
            if seed in visited or diff_ratio[seed] > inner_thr:
                continue

            queue = deque([seed])
            region = set([seed])
            visited.add(seed)
            while queue:
                current = queue.popleft()
                for neighbor in G.neighbors(current):
                    if neighbor in visited or labels[neighbor] != tissue_type:
                        continue

                    if diff_ratio[neighbor] <= inner_thr:
                        visited.add(neighbor)
                        region.add(neighbor)
                        queue.append(neighbor)
                    else:
                        visited.add(neighbor)

            if len(region) < 3:
                continue

            contour = set()
            ext_tissue_counts = defaultdict(int)
            ext_neighbors = set()
            for node in region:
                has_external = False
                for nbr in G.neighbors(node):
                    if nbr not in region:
                        has_external = True
                        ext_neighbors.add(nbr)
                        ext_tissue_counts[labels[nbr]] += 1
                if has_external:
                    contour.add(node)

            contours_dict[region_id] = contour
            surround_counts[region_id] = dict(ext_tissue_counts)
            surround_nodes_dict[region_id] = ext_neighbors

            if not ext_neighbors:
                external_ratio = 1.0
                main_surround = tissue_type
            else:
                main_surround, max_count = max(ext_tissue_counts.items(), key=lambda x: x[1])
                external_ratio = max_count / len(ext_neighbors)

            regions_info.append({
                "region_id": region_id,
                "tissue": tissue_type,
                "size": len(region),
                "contour_size": len(contour),
                "external_ratio": external_ratio,
                "surrounding_tissue": main_surround,
                "surround_count": ext_tissue_counts,
                "nodes": list(region),
                "surround_nodes": list(ext_neighbors)
            })
            region_id += 1


    all_regions = pd.DataFrame(regions_info)
    all_regions["surround_details"] = all_regions["surround_count"].apply(
        lambda x: "; ".join(f"{k}({v})" for k,v in sorted(x.items(), key=lambda i: -i[1]))
    )
    return all_regions


def get_surround_regions_enhanced(all_regions: pd.DataFrame) -> List[Dict]:
    node_to_region = {}
    region_info = {}

    for _, row in all_regions.iterrows():
        for node in row['nodes']:
            node_to_region[node] = row['region_id']
        region_info[row['region_id']] = {
            'tissue': row['tissue'],
            'size': row['size']
        }

    results = []
    for _, target_row in all_regions.iterrows():
        surround_nodes = target_row['surround_nodes']
        counter = defaultdict(int)
        for node in surround_nodes:
            if node in node_to_region:
                counter[node_to_region[node]] += 1

        target_id = target_row['region_id']
        counter.pop(target_id, None)
        results.append({
            "region_id": target_id,
            "region_tissue": target_row['tissue'],
            "surrounded_by": [
                {
                    "region_id": rid,
                    "tissue": region_info[rid]['tissue'],
                    "size": region_info[rid]['size'],
                    "contact_nodes": count
                }
                for rid, count in sorted(counter.items())
            ]
        })

    return results


def viz_region(adata_sc, all_regions, l_idxs, tissue_attr: str = "CAST_label"):
    fig = plt.figure(figsize=(12, 8))
    for idx_region in l_idxs:
        ax = fig.add_subplot(1,2,1)
        legend_loc = None
        if idx_region == l_idxs[-1]:
            legend_loc = "right margin"

        adata_nodes = adata_sc[np.array(list(all_regions.loc[idx_region]["nodes"])), :]
        adata_neighbors = adata_sc[np.array(list(all_regions.loc[idx_region]["surround_nodes"])), :]

        sc.pl.spatial(adata_nodes, color=tissue_attr, spot_size=5, frameon=False, ax=ax, legend_loc=None, show=False)
        sc.pl.spatial(adata_neighbors, color=tissue_attr, spot_size=5, frameon=False, ax=ax, legend_loc=legend_loc, show=False)

        ax = fig.add_subplot(1,2,2)
        sc.pl.spatial(adata_nodes, color="cell_type", spot_size=5, frameon=False, ax=ax, legend_loc=None, show=False)
        sc.pl.spatial(adata_neighbors, color="cell_type", spot_size=5, frameon=False, ax=ax, legend_loc=legend_loc, show=False)

        df_region_surr = get_surround_regions_enhanced(all_regions)
        print(pd.DataFrame(df_region_surr).loc[idx_region, "surrounded_by"])


def compute_region_convex_hulls(adata_sc: sc.AnnData, all_regions: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the convex hull for each region defined in all_regions.
    
    Args:
        adata_sc: AnnData, Containing spatial coordinates in adata_sc.obsm["spatial"]
        all_regions: DataFrame，Containing regions with a 'nodes' column listing cell indices
    
    Returns:
        gdf_hulls: GeoDataFrame. Containing convex hulls and statistics for each region
    """
    hull_records = []
    
    for idx_region in range(len(all_regions)):
        nodes = np.array(list(all_regions.loc[idx_region]["nodes"]))
        coords = adata_sc[nodes, :].obsm["spatial"]
        
        if len(coords) < 3:
            continue
            
        try:
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            
            from shapely.geometry import Polygon
            hull_polygon = Polygon(hull_points)
            hull_records.append({
                'region_id': all_regions.loc[idx_region]['region_id'],
                'size': all_regions.loc[idx_region]['size'],
                'geometry': hull_polygon,
                'area': hull_polygon.area,
                'perimeter': hull_polygon.length,
                'n_vertices': len(hull.vertices)
            })

        except Exception as e:
            print(f"Region {idx_region} convex hull calculation failed: {e}")
            continue

    gdf_hulls = gpd.GeoDataFrame(hull_records)
    return gdf_hulls


def get_cast_regions(
    adata_sc: sc.AnnData,
    tissue_attr: str = "CAST_label",
    inner_thr: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Get cast regions from a spatial network graph.

    Args:
        adata_sc: Annotated data object with spatial coordinates in adata_sc.obsm["spatial"]
        tissue_attr: Node attribute name for tissue type
        inner_thr: Maximum allowed proportion of neighboring nodes with different tissue type during region growing (0-1)

    Returns:
        all_regions: DataFrame with statistics for all regions,
    """
    from CAST import coords2adjacentmat
    
    coords = adata_sc.obsm["spatial"]
    delaunay_graph = coords2adjacentmat(coords, output_mode='raw', strategy_t='convex')

    add_obs_labels_to_graph(delaunay_graph, adata_sc, obs_col=tissue_attr, label_attr=tissue_attr)
    all_regions = find_enclosed_regions(delaunay_graph, tissue_attr=tissue_attr, inner_thr=inner_thr)
    gdf_region_hulls = compute_region_convex_hulls(adata_sc, all_regions)
    
    all_regions = gpd.GeoDataFrame(all_regions)
    all_regions.geometry = gdf_region_hulls.geometry
    all_regions["area"] = gdf_region_hulls["area"]
    region_anno(adata_sc, all_regions, label="tissue")
    region_anno(adata_sc, all_regions, label="cell_type")

    return all_regions


def plot_region_coutour(        
        all_regions: gpd.GeoDataFrame, 
        crop_region: Tuple[int, int, int, int]=None,
        show_text: bool = False,
        color_outline: str = 'red',
        color_hole: str = 'green',
        ax: plt.Axes = None
    ) -> plt.figure:
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

    x_min, x_max, y_min, y_max = 0, 16000, 0, 16000
    intersecting_regions = all_regions
    if crop_region is not None:
        x_min, x_max, y_min, y_max = crop_region
    
    crop_box = box(x_min, y_min, x_max, y_max)
    all_regions = all_regions[all_regions.geometry.intersects(crop_box)]    
    for idx, row in intersecting_regions.iterrows():
        geom = row.geometry
        
        if geom.is_empty or not geom.intersects(crop_box):
            continue

        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            x = x - np.array(crop_box.bounds[0])
            y = y - np.array(crop_box.bounds[1])
            ax.plot(x, y, color=color_outline, linestyle='-', linewidth=2, alpha=0.8)
            
            for interior in geom.interiors:
                x_hole, y_hole = interior.xy
                x_hole = x_hole - np.array(crop_box.bounds[0])
                y_hole = y_hole - np.array(crop_box.bounds[1])
                ax.plot(x_hole, y_hole, color=color_hole, linestyle='--', linewidth=1, alpha=0.6)
                
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                x, y = polygon.exterior.xy
                x = x - np.array(crop_box.bounds[0])
                y = y - np.array(crop_box.bounds[1])
                ax.plot(x, y, color=color_outline, linestyle='-', linewidth=2, alpha=0.8)
                
                for interior in polygon.interiors:
                    x_hole, y_hole = interior.xy
                    x_hole = x_hole - np.array(crop_box.bounds[0])
                    y_hole = y_hole - np.array(crop_box.bounds[1])
                    ax.plot(x_hole, y_hole, color=color_hole, linestyle='--', linewidth=1, alpha=0.6)
        

        centroid = row.geometry.centroid
        if show_text and crop_box.contains(centroid):
            ax.text(centroid.x - x_min, centroid.y - y_min, f"{idx}"
        )
    
    ax.set_xlim(0, crop_box.bounds[2] - crop_box.bounds[0])
    ax.set_ylim(0, crop_box.bounds[3] - crop_box.bounds[1])
    ax.invert_yaxis()  
    return fig


def create_outline(
        gdf: gpd.GeoDataFrame,
        thickness: float=50,
        simplify_tolerance: float=None,
        keep_attributes: bool=True
    ):
    """
    Create an outline of specified thickness for each geometry in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.
        thickness (float): Outline thickness in pixels. Positive for outer, negative for inner outline.
        simplify_tolerance (float, optional): Tolerance for geometry simplification.
        keep_attributes (bool): Whether to keep original attribute columns.

    Returns:
        GeoDataFrame: GeoDataFrame with outline geometries, same number of rows as input.
    """
    import geopandas as gpd

    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    outlines = []

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if simplify_tolerance is not None and hasattr(geom, 'simplify'):
            geom = geom.simplify(simplify_tolerance)

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

def get_cells_in_outline(adata_sc: sc.AnnData, outline_gdf: gpd.GeoDataFrame, predicate: str='within') -> Tuple[sc.AnnData, np.ndarray]:
    """
    Extract cells within a specified outline using spatial join.

    Args:
        adata_sc: AnnData object containing spatial coordinates in adata_sc.obsm["spatial"]
        outline_gdf: GeoDataFrame containing outline geometries
        predicate: Spatial relationship predicate, can be 'within' or 'intersects'

    Returns:
        subset_adata: AnnData object containing only cells within the outline
        cell_indices: Indices of cells within the outline
    """
    from shapely.geometry import Point
    import geopandas as gpd

    coords = adata_sc.obsm["spatial"]
    points_gdf = gpd.GeoDataFrame(
        {'cell_id': range(len(coords))},
        geometry=[Point(x, y) for x, y in coords]
    )
    joined = gpd.sjoin(points_gdf, outline_gdf, how='inner', predicate=predicate)
    
    cell_indices = joined['cell_id'].unique()
    subset_adata = adata_sc[cell_indices].copy()
    return subset_adata, cell_indices

def interaction_plot(gdf_all:gpd.GeoDataFrame, adata_sc: sc.AnnData, key: str="tissue", distance_threshold: int=500):
    """
    Plot interactions between regions based on a distance threshold.

    Args:
        gdf_all: GeoDataFrame containing region geometries and labels
        adata_sc: Annotated data object with spatial coordinates in adata_sc.obsm["spatial"]
        key: Column name for tissue type in gdf_all, default "label_CAST_label"
        distance_threshold: Distance threshold for interactions, default 500
    
    """
    interactions_df, g = analyze_region_interactions(gdf_all, distance_threshold=distance_threshold, key=f"label_{key}")
    
    if len(interactions_df) > 0:
        if f"{key}_colors" in adata_sc.uns:
            tissue_types = g.vs['label']
            unique_tissues = adata_sc.obs[key].cat.categories
            tissue_colors_list = adata_sc.uns[f"{key}_colors"]
            
            node_colors = []
            for tissue in tissue_types:
                if tissue in unique_tissues:
                    idx = list(unique_tissues).index(tissue)
                    node_colors.append(tissue_colors_list[idx])
                else:
                    node_colors.append('lightgray')  
            
            plot_interaction_network(g, interactions_df, node_colors=node_colors)
            plot_spatial_interactions(gdf_all, interactions_df, key=f"label_{key}",
                                distance_threshold=distance_threshold, adata_sc=adata_sc)

    


def analyze_region_interactions(gdf_all, distance_threshold=500, key="label_tissue"):
    """
    Analyze interactions between regions and visualize with igraph
    
    Args:
        gdf_all: GeoDataFrame containing geometry and label_tissue columns
        distance_threshold: Distance threshold, default 500
    
    Returns:
        interactions_df: DataFrame of interactions
        g: igraph graph object
    """
    import igraph as ig
    
    centroids = gdf_all.geometry.centroid
    
    interactions = []
    
    for i in range(len(gdf_all)):
        for j in range(i+1, len(gdf_all)):
            dist = centroids.iloc[i].distance(centroids.iloc[j])
            
            if dist <= distance_threshold:
                tissue_i = gdf_all.iloc[i][key]
                tissue_j = gdf_all.iloc[j][key]
                
                interactions.append({
                    'region_i': i,
                    'region_j': j,
                    'tissue_i': tissue_i,
                    'tissue_j': tissue_j,
                    'distance': dist,
                    'interaction_type': f"{tissue_i}-{tissue_j}",
                    'centroid_i_x': centroids.iloc[i].x,
                    'centroid_i_y': centroids.iloc[i].y,
                    'centroid_j_x': centroids.iloc[j].x,
                    'centroid_j_y': centroids.iloc[j].y,
                })
    
    interactions_df = pd.DataFrame(interactions)
    
    if len(interactions_df) == 0:
        print("No interactions found within the distance threshold")
        return interactions_df, None
    
    interaction_counts = interactions_df['interaction_type'].value_counts()
    
    all_tissues = list(set(interactions_df['tissue_i'].tolist() + interactions_df['tissue_j'].tolist()))
    
    tissue_interactions = defaultdict(int)
    for _, row in interactions_df.iterrows():
        pair = tuple(sorted([row['tissue_i'], row['tissue_j']]))
        tissue_interactions[pair] += 1
    
    edges = []
    edge_weights = []
    for (tissue1, tissue2), count in tissue_interactions.items():
        edges.append((tissue1, tissue2))
        edge_weights.append(count)
    
    g = ig.Graph()
    g.add_vertices(all_tissues)
    g.add_edges(edges)
    g.es['weight'] = edge_weights
    
    g.vs['label'] = all_tissues
    g.vs['size'] = [interactions_df[interactions_df['tissue_i'] == tissue].shape[0] + 
                   interactions_df[interactions_df['tissue_j'] == tissue].shape[0] 
                   for tissue in all_tissues]
    
    g.es['width'] = [w * 2 for w in edge_weights]
    
    return interactions_df, g


def plot_interaction_network(g, interactions_df, key="cell_type", figsize=(10, 8), node_colors=None):
    """
    Plot interaction network graph
    
    Args:
        g: igraph graph object
        interactions_df: DataFrame of interactions
        figsize: Figure size tuple
        node_colors: List of colors for nodes, or None for default colors
    """

    if g is None:
        print("No network to plot")
        return
    
    layout = g.layout("fruchterman_reingold")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    node_sizes = []
    for tissue in g.vs['label']:
        tissue_count = len(interactions_df[
            (interactions_df[f'{key}_i'] == tissue) | 
            (interactions_df[f'{key}_j'] == tissue)
        ])
        node_sizes.append(tissue_count)
    
    min_size, max_size = 20, 100
    if max(node_sizes) > 0:
        normalized_sizes = [
            min_size + (size / max(node_sizes)) * (max_size - min_size) 
            for size in node_sizes
        ]
    else:
        normalized_sizes = [min_size] * len(node_sizes)
    
    if len(g.es['weight']) > 0:
        max_weight = max(g.es['weight'])
        edge_widths = [1 + (w / max_weight) * 4 for w in g.es['weight']]  
    else:
        edge_widths = [1] * len(g.es)
    
    if node_colors is not None:
        if len(node_colors) != len(g.vs):
            print(f"Warning: node_colors length ({len(node_colors)}) doesn't match number of nodes ({len(g.vs)})")
            vertex_colors = 'lightblue'  
        else:
            vertex_colors = node_colors
    else:
        vertex_colors = 'lightblue'  
    
    ig.plot(g, 
            layout=layout,
            vertex_label=g.vs['label'],
            vertex_size=normalized_sizes,
            edge_width=edge_widths,
            vertex_color=vertex_colors,
            vertex_label_size=10,
            vertex_frame_color='black',
            vertex_frame_width=1,
            edge_color='gray',
            target=ax1,
            bbox=(0, 0, 400, 400),
            margin=50)
    
    ax1.set_title("Tissue Interaction Network", fontsize=12, fontweight='bold')
    
    interaction_counts = interactions_df['cell_interaction_type'].value_counts().head(10)  
    ax2.barh(range(len(interaction_counts)), interaction_counts.values, color='steelblue')
    ax2.set_yticks(range(len(interaction_counts)))
    ax2.set_yticklabels(interaction_counts.index, fontsize=9)
    ax2.set_xlabel("Number of Interactions")
    ax2.set_title("Top 10 Interaction Types")
    ax2.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(interaction_counts.values):
        ax2.text(v + 0.1, i, str(v), va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_spatial_interactions(gdf_all, interactions_df, key="label_tissue", distance_threshold=500, tissue_colors=None, adata_sc=None):
    """
    Display interactions on spatial map
    
    Args:
        gdf_all: GeoDataFrame containing regions
        interactions_df: DataFrame of interactions
        key: Column name for tissue type
        distance_threshold: Distance threshold for interactions
        tissue_colors: Dict mapping tissue types to colors, or None to use default
        adata_sc: AnnData object to get colors from uns, optional
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_tissues = gdf_all[key].unique()
    
    if tissue_colors is not None:
        colors_dict = tissue_colors
    elif adata_sc is not None and f"{key.replace('label_', '')}_colors" in adata_sc.uns:
        tissue_key = key.replace('label_', '')
        tissue_categories = adata_sc.obs[tissue_key].cat.categories
        tissue_colors_list = adata_sc.uns[f"{tissue_key}_colors"]
        colors_dict = dict(zip(tissue_categories, tissue_colors_list))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_tissues)))
        colors_dict = dict(zip(unique_tissues, colors))
    
    for tissue in unique_tissues:
        mask = gdf_all[key] == tissue
        color = colors_dict.get(tissue, 'lightgray')  
        
        gdf_all[mask].plot(
            ax=ax, 
            color=color, 
            alpha=0.7, 
            edgecolor='black', 
            linewidth=0.5,
            label=tissue
        )
    
    for _, row in interactions_df.iterrows():
        ax.plot([row['centroid_i_x'], row['centroid_j_x']], 
                [row['centroid_i_y'], row['centroid_j_y']], 
                'r-', alpha=0.5, linewidth=1)

    ax.set_title(f"Spatial interaction (Distance threshold: {distance_threshold})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return fig


def find_knn_between_groups(coords_A, coords_B, k=5, distance_threshold=None):
    """
    Find k-nearest neighbors between two groups of coordinates.
    
    Args:
        coords_A: coords A [N_A, 2]
        coords_B: coords B [N_B, 2]  
        k: k for knn
        distance_threshold: distance threshold to filter edges
    
    Returns:
        edges: List of tuples (idx_A, idx_B, distance)
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(coords_A) == 0 or len(coords_B) == 0:
        return []
    
    nbrs = NearestNeighbors(n_neighbors=min(k, len(coords_B)), algorithm='kd_tree')
    nbrs.fit(coords_B)
    
    distances, indices = nbrs.kneighbors(coords_A)
    
    edges = []
    for i_A in range(len(coords_A)):
        for j, (dist, i_B) in enumerate(zip(distances[i_A], indices[i_A])):
            if distance_threshold is None or dist <= distance_threshold:
                edges.append((i_A, i_B, dist))
    
    return edges


def analyze_cell_interactions(gdf_all, adata_sc, distance_threshold=500, cell_distance_threshold=50, k_neighbors=5, key="label_tissue"):
    """
    Analysis cell-cell interactions based on region distances and cell coordinates.
    
    Args:
        gdf_all: GeoDataFrame containing regions with nodes information
        adata_sc: AnnData object with cell coordinates
        distance_threshold: threshold for region distances
        cell_distance_threshold: threshold for cell distances
        k_neighbors: number of neighbors to consider
        key: column name for tissue type in gdf_all
        
    Returns:
        cell_interactions_df: cell interaction pd.DataFrame
        g_cells: igraph for cell-cell interactions
        region_pairs: region pairs within distance threshold
    """
    import igraph as ig
    from itertools import combinations
    
    centroids = gdf_all.geometry.centroid
    region_pairs = []
    
    for i in range(len(gdf_all)):
        for j in range(i+1, len(gdf_all)):
            dist = centroids.iloc[i].distance(centroids.iloc[j])
            if dist <= distance_threshold:
                region_pairs.append({
                    'region_i': i,
                    'region_j': j, 
                    'tissue_i': gdf_all.iloc[i][key],
                    'tissue_j': gdf_all.iloc[j][key],
                    'region_distance': dist
                })
    
    if len(region_pairs) == 0:
        print("No region pairs found within distance threshold")
        return pd.DataFrame(), None, []
    
    print(f"Found {len(region_pairs)} region pairs within {distance_threshold} distance")
    
    coords = adata_sc.obsm["spatial"]
    
    all_cell_edges = []
    cell_to_region = {}  
    
    for pair_idx, pair in tqdm(enumerate(region_pairs)):
        nodes_i = gdf_all.iloc[pair['region_i']]['nodes']
        nodes_j = gdf_all.iloc[pair['region_j']]['nodes']
        
        if len(nodes_i) == 0 or len(nodes_j) == 0:
            continue
            
        for node in nodes_i:
            cell_to_region[node] = pair['region_i']
        for node in nodes_j:
            cell_to_region[node] = pair['region_j']
        
        coords_i = coords[nodes_i]
        coords_j = coords[nodes_j]
        
        edges_i_to_j = find_knn_between_groups(coords_i, coords_j, k_neighbors, cell_distance_threshold)
        edges_j_to_i = find_knn_between_groups(coords_j, coords_i, k_neighbors, cell_distance_threshold)
        
        for i_local, j_local, dist in edges_i_to_j:
            cell_i = nodes_i[i_local]
            cell_j = nodes_j[j_local]
            all_cell_edges.append({
                'cell_i': cell_i,
                'cell_j': cell_j,
                'distance': dist,
                'region_i': pair['region_i'],
                'region_j': pair['region_j'],
                'tissue_i': pair['tissue_i'],
                'tissue_j': pair['tissue_j'],
                'interaction_type': f"{pair['tissue_i']}-{pair['tissue_j']}",
                'cell_type_i': adata_sc.obs.iloc[cell_i]['cell_type'],
                'cell_type_j': adata_sc.obs.iloc[cell_j]['cell_type'],
                'cell_interaction_type': f"{adata_sc.obs.iloc[cell_i]['cell_type']}-{adata_sc.obs.iloc[cell_j]['cell_type']}"
            })
        
        for j_local, i_local, dist in edges_j_to_i:
            cell_j = nodes_j[j_local] 
            cell_i = nodes_i[i_local]
            existing = any(edge['cell_i'] == cell_i and edge['cell_j'] == cell_j for edge in all_cell_edges)
            if not existing:
                all_cell_edges.append({
                    'cell_i': cell_i,
                    'cell_j': cell_j,
                    'distance': dist,
                    'region_i': pair['region_j'],  
                    'region_j': pair['region_i'],
                    'tissue_i': pair['tissue_j'],
                    'tissue_j': pair['tissue_i'], 
                    'interaction_type': f"{pair['tissue_j']}-{pair['tissue_i']}",
                    'cell_type_i': adata_sc.obs.iloc[cell_i]['cell_type'],
                    'cell_type_j': adata_sc.obs.iloc[cell_j]['cell_type'],
                    'cell_interaction_type': f"{adata_sc.obs.iloc[cell_i]['cell_type']}-{adata_sc.obs.iloc[cell_j]['cell_type']}"
                })
    
    cell_interactions_df = pd.DataFrame(all_cell_edges)
    
    if len(cell_interactions_df) == 0:
        print("No cell interactions found within thresholds")
        return cell_interactions_df, None, region_pairs
    
    print(f"Found {len(cell_interactions_df)} cell-cell interactions")
    
    all_interacting_cells = list(set(cell_interactions_df['cell_i'].tolist() + cell_interactions_df['cell_j'].tolist()))
    
    cell_to_graph_idx = {cell_id: idx for idx, cell_id in enumerate(all_interacting_cells)}
    
    edges = []
    edge_weights = []
    for _, row in cell_interactions_df.iterrows():
        i = cell_to_graph_idx[row['cell_i']]
        j = cell_to_graph_idx[row['cell_j']]
        edges.append((i, j))
        edge_weights.append(1.0 / (row['distance'] + 1))  
    
    g_cells = ig.Graph()
    g_cells.add_vertices(len(all_interacting_cells))
    g_cells.add_edges(edges)
    g_cells.es['weight'] = edge_weights
    g_cells.es['distance'] = cell_interactions_df['distance'].tolist()
    
    g_cells.vs['cell_id'] = all_interacting_cells
    g_cells.vs['cell_type'] = [adata_sc.obs.iloc[cell_id]['cell_type'] for cell_id in all_interacting_cells]
    g_cells.vs['tissue'] = [adata_sc.obs.iloc[cell_id].get('tissue', 'Unknown') for cell_id in all_interacting_cells]
    g_cells.vs['region_id'] = [cell_to_region.get(cell_id, -1) for cell_id in all_interacting_cells]
    
    cell_coords = coords[all_interacting_cells]
    g_cells.vs['x'] = cell_coords[:, 0].tolist()
    g_cells.vs['y'] = cell_coords[:, 1].tolist()
    
    return cell_interactions_df, g_cells, region_pairs


def plot_cell_interaction_network(g_cells, cell_interactions_df, figsize=(15, 5), node_colors=None):
    """
    Draw cell interaction network
    
    Args:
        g_cells: igraph object for cell interactions
        cell_interactions_df: DataFrame of cell interactions
        figsize: Figure size
        node_colors: colors for nodes
    """
    import matplotlib.pyplot as plt
    import igraph as ig
    
    if g_cells is None:
        print("No cell network to plot")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    if len(g_cells.vs) > 0:
        coords = np.array([[v['x'], v['y']] for v in g_cells.vs])
        coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        layout = coords_norm * 400  
        
        degrees = g_cells.degree()
        node_sizes = [max(5, min(20, d * 2)) for d in degrees]
        
        edge_weights = g_cells.es['weight']
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + (w / max_weight) * 2 for w in edge_weights]
        
        ig.plot(g_cells,
                layout=layout,
                vertex_size=node_sizes,
                vertex_color=node_colors if node_colors else 'lightblue',
                vertex_frame_color='black',
                vertex_frame_width=0.5,
                edge_width=edge_widths,
                edge_color='gray',
                target=ax1,
                bbox=(0, 0, 400, 400),
                margin=20)
    
    ax1.set_title("Cell Interaction Network")
    
    cell_interaction_counts = cell_interactions_df['cell_interaction_type'].value_counts().head(10)
    ax2.barh(range(len(cell_interaction_counts)), cell_interaction_counts.values, color='steelblue')
    ax2.set_yticks(range(len(cell_interaction_counts)))
    ax2.set_yticklabels(cell_interaction_counts.index, fontsize=8)
    ax2.set_xlabel("Number of Interactions")
    ax2.set_title("Top 10 Cell Type Interactions")
    ax2.grid(axis='x', alpha=0.3)
    
    ax3.hist(cell_interactions_df['distance'], bins=30, alpha=0.7, color='lightgreen')
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Cell Interaction Distance Distribution")
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_cell_interaction_network_fast(g_cells, cell_interactions_df, figsize=(15, 5), node_colors=None, max_edges_to_plot=5000):
    """
    Draw cell interaction network with edge sampling to improve performance
    
    Args:
        g_cells: igraph object for cell interactions
        cell_interactions_df: cell-cell interaction pd.DataFrame
        figsize: figure size
        node_colors: node colors
        max_edges_to_plot: Maximum number of edges to plot to avoid clutter
    """
    import matplotlib.pyplot as plt
    import igraph as ig
    
    if g_cells is None:
        print("No cell network to plot")
        return
    
    if len(g_cells.es) > max_edges_to_plot:
        edge_sample = np.random.choice(len(g_cells.es), max_edges_to_plot, replace=False)
        g_sample = g_cells.copy()
        edges_to_delete = [i for i in range(len(g_cells.es)) if i not in edge_sample]
        g_sample.delete_edges(edges_to_delete)
        
        isolated = [v.index for v in g_sample.vs if v.degree() == 0]
        g_sample.delete_vertices(isolated)
        g_plot = g_sample
    else:
        g_plot = g_cells
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    if len(g_plot.vs) > 0:
        coords = np.array([[v['x'], v['y']] for v in g_plot.vs])
        coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
        layout = coords_norm * 400
        
        degrees = g_plot.degree()
        node_sizes = [max(2, min(15, d)) for d in degrees]
        
        edge_weights = g_plot.es['weight'] if 'weight' in g_plot.es.attributes() else [1] * len(g_plot.es)
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.2 + (w / max_weight) * 1 for w in edge_weights]
        
        ig.plot(g_plot,
                layout=layout,
                vertex_size=node_sizes,
                vertex_color=node_colors[:len(g_plot.vs)] if node_colors else 'lightblue',
                vertex_frame_color='black',
                vertex_frame_width=0.3,
                edge_width=edge_widths,
                edge_color='gray',
                target=ax1,
                bbox=(0, 0, 400, 400),
                margin=10)
    
    cell_interaction_counts = cell_interactions_df['cell_interaction_type'].value_counts().head(10)
    ax2.barh(range(len(cell_interaction_counts)), cell_interaction_counts.values, color='steelblue')
    ax2.set_yticks(range(len(cell_interaction_counts)))
    ax2.set_yticklabels(cell_interaction_counts.index, fontsize=8)
    ax2.set_xlabel("Number of Interactions")
    ax2.set_title("Top 10 Cell Type Interactions")
    ax2.grid(axis='x', alpha=0.3)
    
    ax3.hist(cell_interactions_df['distance'], bins=30, alpha=0.7, color='lightgreen')
    ax3.set_xlabel("Distance")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Cell Interaction Distance Distribution")
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_cell_interactions_spatial(gdf_all, cell_interactions_df, adata_sc, key="label_tissue", 
                                 sample_edges=1000, figsize=(12, 10), crop_region=None):
    """
    Display cell interactions on a spatial map (supports local cropping)

    Args:
        gdf_all: GeoDataFrame of regions
        cell_interactions_df: DataFrame containing cell interaction edges (with x_i, y_i, x_j, y_j)
        adata_sc: AnnData object with spatial coordinates
        key: Column name for region label in gdf_all
        sample_edges: Number of edges to sample for plotting to avoid clutter
        figsize: Figure size tuple
        crop_region: Local region tuple (x_min, x_max, y_min, y_max); None to display all
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if crop_region is not None:
        x_min, x_max, y_min, y_max = crop_region
        mask = (
            (cell_interactions_df['x_i'] >= x_min) & (cell_interactions_df['x_i'] <= x_max) &
            (cell_interactions_df['y_i'] >= y_min) & (cell_interactions_df['y_i'] <= y_max) &
            (cell_interactions_df['x_j'] >= x_min) & (cell_interactions_df['x_j'] <= x_max) &
            (cell_interactions_df['y_j'] >= y_min) & (cell_interactions_df['y_j'] <= y_max)
        )
        filtered_df = cell_interactions_df[mask].copy()
        
        region_mask = gdf_all.geometry.bounds.apply(
            lambda bounds: not (bounds['maxx'] < x_min or bounds['minx'] > x_max or 
                              bounds['maxy'] < y_min or bounds['miny'] > y_max), axis=1
        )
        gdf_filtered = gdf_all[region_mask].copy()
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        title_suffix = f" (cropped region)"
    else:
        filtered_df = cell_interactions_df
        gdf_filtered = gdf_all
        title_suffix = ""
    
    unique_tissues = gdf_filtered[key].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_tissues)))
    color_dict = dict(zip(unique_tissues, colors))
    
    for tissue in unique_tissues:
        mask = gdf_filtered[key] == tissue
        gdf_filtered[mask].plot(ax=ax, color=color_dict[tissue], alpha=0.3, 
                              edgecolor='black', linewidth=0.5, label=tissue)
    
    if len(filtered_df) > sample_edges:
        sampled_df = filtered_df.sample(n=sample_edges, random_state=42)
    else:
        sampled_df = filtered_df
    
    for _, row in sampled_df.iterrows():
        ax.plot([row['x_i'], row['x_j']], [row['y_i'], row['y_j']], 
                'r-', alpha=0.3, linewidth=0.5)
    
    ax.set_title(f"Cell Interactions in Space (showing {len(sampled_df)} of {len(filtered_df)} interactions){title_suffix}")
    if crop_region is None:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def build_faiss_index(coords, use_gpu=False):
    """
    Build a FAISS index for fast nearest neighbor search.
    
    Args:
        coords: [N, 2] cell coordinates
        use_gpu: need use GPU for FAISS index building
    
    Returns:
        index: faiss index object
    """
    import faiss
    
    coords_f32 = coords.astype(np.float32)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, coords_f32.shape[1])
    else:
        index = faiss.IndexFlatL2(coords_f32.shape[1])
    
    index.add(coords_f32)
    
    return index


def faiss_knn_between_groups(faiss_index, coords, query_indices, target_indices, k=5, distance_threshold=None):
    """
    Use FAISS to quickly find k-nearest neighbors between two groups of cells.

    Args:
        faiss_index: Pre-built FAISS index for all cell coordinates.
        coords: All cell coordinates (numpy array).
        query_indices: Indices of query group cells.
        target_indices: Indices of target group cells.
        k: Number of neighbors to find for each query cell.
        distance_threshold: Optional distance threshold.

    Returns:
        edges: List of tuples (query_idx, target_idx, distance)
    """
    if len(query_indices) == 0 or len(target_indices) == 0:
        return []
    
    query_coords = coords[query_indices].astype(np.float32)
    
    search_k = min(len(coords), k * 10)  
    distances, indices = faiss_index.search(query_coords, search_k)
    
    target_set = set(target_indices)
    
    edges = []
    for i, query_idx in enumerate(query_indices):
        found_count = 0
        for j in range(search_k):
            if found_count >= k:
                break
                
            neighbor_idx = indices[i, j]
            distance = np.sqrt(distances[i, j])  
            
            if neighbor_idx in target_set:
                if distance_threshold is None or distance <= distance_threshold:
                    edges.append((query_idx, neighbor_idx, distance))
                    found_count += 1
    
    return edges


def analyze_region_cell_edge_interactions(
        gdf_all: gpd.GeoDataFrame,
        adata_sc: sc.AnnData,
        distance_threshold: int=50,
        cell_distance_threshold: int=50,
        k_neighbors: int=5,
        key: str="label_tissue",
        use_gpu:bool =False):
    """
    Analyze cell interactions based on region distances and cell coordinates using FAISS.
    
    Args:
        gdf_all: GeoDataFrame containing regions with nodes information
        adata_sc: AnnData object with cell coordinates
        distance_threshold: Threshold for region distances
        cell_distance_threshold: Threshold for cell distances
        k_neighbors: number of neighbors to find for each cell
        key: column name for tissue type in gdf_all
        use_gpu: if use GPU for FAISS index building
        
    Returns:
        cell_interactions_df: DataFrame for cell interactions
        g_cells: igraph object for cell interactions
        region_pairs: region of pairs for information
    """
    import igraph as ig
    from tqdm import tqdm
    from shapely.geometry import Point
    
    coords = adata_sc.obsm["spatial"]
    
    print("Finding cells near region boundaries...")
    region_cell_pairs = []
    
    for region_idx in tqdm(range(len(gdf_all)), desc="Processing regions"):
        region = gdf_all.iloc[region_idx]
        region_geom = region.geometry
        
        if not region_geom.is_valid:
            region_geom = region_geom.buffer(0)
            if not region_geom.is_valid:
                print(f"Warning: Region {region_idx} geometry is invalid and cannot be fixed.")
                continue
                
        minx, miny, maxx, maxy = region_geom.bounds
        mask_bbox = (
            (coords[:, 0] >= minx - distance_threshold) & 
            (coords[:, 0] <= maxx + distance_threshold) & 
            (coords[:, 1] >= miny - distance_threshold) & 
            (coords[:, 1] <= maxy + distance_threshold)
        )
        
        if not mask_bbox.any():
            continue
            
        cell_indices = np.where(mask_bbox)[0]
        filtered_coords = coords[mask_bbox]
        
        for i, (cell_idx, (x, y)) in enumerate(zip(cell_indices, filtered_coords)):
            point = Point(x, y)
            dist = region_geom.distance(point)
            
            if dist <= distance_threshold:
                region_cell_pairs.append({
                    'region_idx': region_idx, 
                    'cell_idx': cell_idx,
                    'distance': dist,
                    'tissue': region[key],
                    'cell_type': adata_sc.obs.iloc[cell_idx]['cell_type'],
                    'x': x,
                    'y': y
                })
    
    if len(region_cell_pairs) == 0:
        print("No cells found near region boundaries")
        return pd.DataFrame(), None, []
        
    print(f"Found {len(region_cell_pairs)} cells near region boundaries")
    
    region_cell_df = pd.DataFrame(region_cell_pairs)
    
    print("Building FAISS index for cell-cell interactions...")
    faiss_index = build_faiss_index(coords, use_gpu=use_gpu)
    
    cell_indices = region_cell_df['cell_idx'].unique()
    
    print("Finding cell-cell interactions...")
    all_cell_edges = []
    
    for cell_idx in tqdm(cell_indices, desc="Processing cell interactions"):
        neighbors_idx, distances = find_nearest_cells_faiss(
            faiss_index, coords, cell_idx, k_neighbors, cell_distance_threshold
        )
        
        cell_regions = region_cell_df[region_cell_df['cell_idx'] == cell_idx]['region_idx'].unique()
        
        for neighbor_idx, dist in zip(neighbors_idx, distances):
            if neighbor_idx == cell_idx:
                continue
                
            neighbor_regions = region_cell_df[region_cell_df['cell_idx'] == neighbor_idx]['region_idx'].unique()
            
            all_cell_edges.append({
                'cell_i': cell_idx,
                'cell_j': neighbor_idx,
                'distance': dist,
                'cell_type_i': adata_sc.obs.iloc[cell_idx]['cell_type'],
                'cell_type_j': adata_sc.obs.iloc[neighbor_idx]['cell_type'],
                'cell_interaction_type': f"{adata_sc.obs.iloc[cell_idx]['cell_type']}-{adata_sc.obs.iloc[neighbor_idx]['cell_type']}",
                'x_i': coords[cell_idx, 0],
                'y_i': coords[cell_idx, 1],
                'x_j': coords[neighbor_idx, 0],
                'y_j': coords[neighbor_idx, 1],
                'regions_i': cell_regions.tolist() if len(cell_regions) > 0 else [],
                'regions_j': neighbor_regions.tolist() if len(neighbor_regions) > 0 else []
            })
    
    cell_interactions_df = pd.DataFrame(all_cell_edges)
    
    if len(cell_interactions_df) == 0:
        print("No cell interactions found")
        return cell_interactions_df, None, region_cell_pairs
    
    print(f"Found {len(cell_interactions_df)} cell-cell interactions")
    
    print("Building cell interaction network...")
    all_interacting_cells = list(set(cell_interactions_df['cell_i'].tolist() + cell_interactions_df['cell_j'].tolist()))
    cell_to_graph_idx = {cell_id: idx for idx, cell_id in enumerate(all_interacting_cells)}
    
    edges = []
    edge_weights = []
    for _, row in cell_interactions_df.iterrows():
        i = cell_to_graph_idx[row['cell_i']]
        j = cell_to_graph_idx[row['cell_j']]
        edges.append((i, j))
        edge_weights.append(1.0 / (row['distance'] + 1))
    
    g_cells = ig.Graph()
    g_cells.add_vertices(len(all_interacting_cells))
    g_cells.add_edges(edges)
    g_cells.es['weight'] = edge_weights
    g_cells.es['distance'] = cell_interactions_df['distance'].tolist()
    
    g_cells.vs['cell_id'] = all_interacting_cells
    g_cells.vs['cell_type'] = [adata_sc.obs.iloc[cell_id]['cell_type'] for cell_id in all_interacting_cells]
    g_cells.vs['x'] = [coords[cell_id, 0] for cell_id in all_interacting_cells]
    g_cells.vs['y'] = [coords[cell_id, 1] for cell_id in all_interacting_cells]
    
    print("Analysis completed!")
    return cell_interactions_df, g_cells, region_cell_df


def find_nearest_cells_faiss(faiss_index, coords, cell_idx, k_neighbors=5, distance_threshold=None):
    """
    Find k nearest cells to a target cell using FAISS index.

    Args:
        faiss_index: Faiss index object
        coords: All cell coordinates
        cell_idx: Target cell index
        k_neighbors: Number of neighbors to find
        distance_threshold: Distance threshold

    Returns:
        neighbors_idx: Indices of neighbor cells
        distances: Corresponding distances
    """
    query = coords[cell_idx].reshape(1, -1).astype(np.float32)
    
    k_search = min(k_neighbors + 1, len(coords))
    distances, indices = faiss_index.search(query, k_search)
    
    distances = np.sqrt(distances[0])  
    indices = indices[0]
    
    if distance_threshold is not None:
        mask = distances <= distance_threshold
        distances = distances[mask]
        indices = indices[mask]
    
    return indices, distances


def _process_single_contour(args):
    """Helper function for multiprocessing."""
    idx_line, adata_sc, gdf_km = args
    gdf_km_sub = gdf_km.iloc[idx_line:idx_line+1]
    subset_adata, cell_indices = get_cells_in_outline(adata_sc, gdf_km_sub)
    return subset_adata.X.mean(0)

def reduce_adata_via_contour(adata_sc, key="kmeans_10", radius=32, n_jobs=None):
    gdf_km = adata_to_regions(adata_sc, key, radius=radius)

    task_args = [(idx, adata_sc, gdf_km) for idx in range(gdf_km.shape[0])]

    if n_jobs is None:
        n_jobs = min(cpu_count(), len(task_args))

    print(f"Using {n_jobs} processes for contour reduction...")

    with Pool(n_jobs) as pool:
        l_arrays = list(tqdm(pool.imap(_process_single_contour, task_args), total=len(task_args)))

    np_reduced = np.stack(l_arrays, axis=0)
    adata_contour = sc.AnnData(np_reduced, obs=gdf_km)
    adata_contour.obs[f"{key}_num"] = adata_contour.obs[f"{key}_num"] - 1
    return adata_contour



def group_avg_adatas(adata, agg_cols=None, window_size=256):
    """
    Divide adata into grid patches of size window_size based on obsm['spatial'] coordinates,
    average the expression of cells within each non-empty patch, and return a new AnnData.

    Args:
        adata : anndata.AnnData
            Input object containing obsm['spatial'] (n_obs x 2), columns [x, y] by default.
        window_size : int or float, default=256
            Patch edge length (same unit as spatial coordinates).

    Returns:
        adata_grouped : anndata.AnnData
            n_obs = number of non-empty patches; X is the average expression per patch (float32).
            .obs includes:
                - patch_id: global patch index (row-major, y then x)
                - x_bin, y_bin: grid coordinates of the patch
                - x0, y0, x1, y1: patch boundaries ([x0,x1), [y0,y1))
                - x_center, y_center: patch center coordinates
                - n_cells: number of cells in the patch
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found. Please ensure it contains 2D coordinates.")
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] should be (n_obs, 2)")

    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    n_x = max(1, int(np.ceil((x_max - x_min) / float(window_size))))
    n_y = max(1, int(np.ceil((y_max - y_min) / float(window_size))))
    ix = np.floor((x - x_min) / float(window_size)).astype(int)
    iy = np.floor((y - y_min) / float(window_size)).astype(int)

    ix = np.clip(ix, 0, n_x - 1)
    iy = np.clip(iy, 0, n_y - 1)

    patch_id = iy * n_x + ix
    unique_pids, inv = np.unique(patch_id, return_inverse=True)
    groups = [[] for _ in range(len(unique_pids))]
    for cell_idx, g in enumerate(inv):
        groups[g].append(cell_idx)

    X = adata.X
    is_sparse = sparse.issparse(X)
    rows = []
    obs_rows = []

    for k, cell_indices in enumerate(groups):
        if not cell_indices:
            continue

        if is_sparse:
            sub = X[cell_indices]
            mean_row = (sub.sum(axis=0) / len(cell_indices))
            mean_row = np.asarray(mean_row).ravel()
        else:
            sub = X[cell_indices, :]
            mean_row = np.asarray(sub, dtype=float).mean(axis=0)

        rows.append(mean_row.astype(np.float32, copy=False))

        pid = int(unique_pids[k])
        xb = pid % n_x
        yb = pid // n_x

        x0 = x_min + xb * float(window_size)
        x1 = x0 + float(window_size)
        y0 = y_min + yb * float(window_size)
        y1 = y0 + float(window_size)

        dict_line = {
            "patch_id": pid,
            "x_bin": xb,
            "y_bin": yb,
            "x0": x0, "x1": x1,
            "y0": y0, "y1": y1,
            "x_center": (x0 + x1) / 2.0,
            "y_center": (y0 + y1) / 2.0,
            "n_cells": len(cell_indices),
        }
        agg_info = None
        if agg_cols is not None:
            if type(agg_cols) is str:
                agg_cols = [agg_cols]

            for agg_col in agg_cols:
                agg_info = adata.obs.iloc[cell_indices][agg_cols]
                agg_mode = agg_info.mode().values.ravel()[0] if not agg_info.empty else None
                dict_line[agg_col] = agg_mode

        obs_rows.append(dict_line)

    X_grouped = np.vstack(rows) if rows else np.zeros((0, adata.n_vars), dtype=np.float32)
    obs_df = pd.DataFrame(obs_rows)
    var = adata.var.copy()
    adata_grouped = sc.AnnData(X_grouped, obs=obs_df, var=var)
    adata_grouped.uns["grid_info"] = {
        "window_size": float(window_size),
        "n_x": int(n_x),
        "n_y": int(n_y),
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "patch_id_scheme": "row_major (pid = y_bin * n_x + x_bin)",
    }
    adata_grouped.obsm["spatial"] = adata_grouped.obs[["x_center", "y_center"]].values
    return adata_grouped
