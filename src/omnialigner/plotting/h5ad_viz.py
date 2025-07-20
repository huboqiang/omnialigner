from typing import List

import scanpy as sc
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon
import pandas as pd
import geopandas as gpd
import igraph as ig

from omnialigner.dtypes import Np_image_HWC, Np_image_Mask
from omnialigner.plotting.matplotlib_init import plt

def gdf_shape_to_image(gdf: gpd.GeoDataFrame, key: str="clusters", w: int=1280, h: int=1280) -> Np_image_Mask:
    minx, miny, maxx, maxy = 0, 0, w, h

    x_res = w
    y_res = h
    pixel_size_x = (maxx - minx) / x_res
    pixel_size_y = (maxy - miny) / y_res

    raster_matrix = np.zeros((y_res, x_res), dtype=np.int32)

    transform = rasterio.transform.from_origin(minx, maxy, pixel_size_x, pixel_size_y)

    shapes = []
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if isinstance(geometry, Point):
            buffered_geom = geometry.buffer(row.radius)
            shapes.append((buffered_geom, row[key]))
        elif isinstance(geometry, Polygon):
            shapes.append((geometry, row[key]))
        else:
            print("Cannot handle geometry type:", type(geometry), row, geometry)

    rasterized = rasterize(
        shapes,
        out_shape=raster_matrix.shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='int32'
    )
    raster_matrix[:, :] = rasterized[::-1, :]
    return raster_matrix


def adata_to_gpd(adata: sc.AnnData, key: str="clusters", radius: int=2):
    pd_coords = adata.obsm["spatial"]
    points = [Point(x, y) for x, y in zip(pd_coords[:,0], pd_coords[:,1])]
    gdf = gpd.GeoDataFrame(geometry=points)
    try:
        gdf[key] = adata.obs[key].values.astype(np.int32)+1
    except:
        gdf[key] = adata.obs[key].cat.codes.astype(np.int32).values+1
        
    gdf["radius"] = radius
    return gdf

def compute_label_features(np_exp: Np_image_HWC, np_mask: Np_image_Mask) -> np.ndarray:
    """
    Compute the average features for each label in the mask.
    This function takes a feature map and a mask, and computes the average feature vector for each unique label in the mask.

    Args:
        np_exp: [H, W, C] feature map
        np_mask: [H, W] int32 labels（from 1 to n）

    Returns:
        features: [n, C] average feature vector for each label
    """
    H, W, C = np_exp.shape
    labels = np_mask.flatten()               # [H*W]
    features = np_exp.reshape(-1, C)         # [H*W, C]
    n = labels.max()

    sums = np.zeros((n, C), dtype=np.float64)
    for c in range(C):
        sums[:, c] = np.bincount(labels, weights=features[:, c], minlength=n+1)[1:]

    counts = np.bincount(labels, minlength=n+1)[1:]  # [n]
    counts = np.maximum(counts, 1)             # avoid division by zero
    avg_features = sums / counts[:, None]      # [n, C]
    return avg_features


def plot_interaction_network(g: ig.Graph, interactions_df: pd.DataFrame, key: str="cell_type", n_top:int=10, figsize=(10, 8), node_colors: List[str]=None):
    """
    Plot interaction network graph
    
    Args:
        g: igraph graph object
        interactions_df: DataFrame of interactions
        key: Column name in interactions_df to use for node labels
        n_top: Number of top interaction types to display in the bar chart
        figsize: Figure size tuple
        node_colors: List of colors for nodes, or None for default colors
    """

    if g is None:
        print("No network to plot")
        return
    
    # Set layout
    layout = g.layout("fruchterman_reingold")
    
    # Create figure
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
    interaction_counts = interactions_df['cell_interaction_type'].value_counts().head(n_top)
    ax2.barh(range(len(interaction_counts)), interaction_counts.values, color='steelblue')
    ax2.set_yticks(range(len(interaction_counts)))
    ax2.set_yticklabels(interaction_counts.index, fontsize=9)
    ax2.set_xlabel("Number of Interactions")
    ax2.set_title("Top 10 Interaction Types")
    ax2.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(interaction_counts.values):
        ax2.text(v + 0.1, i, str(v), va='center', fontsize=8)
    
    plt.close()
    return fig