from typing import Tuple

import numpy as np
import dask.array as da
import pandas as pd
import scanpy as sc
from matplotlib.colors import ListedColormap

import omnialigner as om
import matplotlib

plt = om.pl.plt

leaf_color_map = {
    # ── Lymphoid ─────────────────────────
    "B Cell"          : "#004488",
    "Tc"              : "#FF0000",
    "Th"              : "#FF6666",
    "Treg"            : "#FF66B3",
    "DNT"             : "#8B2500",
    "NK"              : "#0066BB",

    # ── Myeloid ─────────────────────────
    "CD163+ Mac"      : "#FFD700",
    "CD163- Mac"      : "#009E60",
    "Monocyte"        : "#7DB317",
    "Dendritic Cell"  : "#B03AAC",
    "Neu"             : "#44C2A5",

    # ── Stroma / Structure ──────────────
    "Epithelial"      : "#8B4513",
    "Cancer cell"     : "#00D4FF",
    "Fibroblast"      : "#FFAA00",
    "Vascular"        : "#BB6600",
    "Endothelial"     : "#FFC0CB",
    "Mast cell"       : "#A1DB00",
    "Undefined"       : "#EE7700",
}


tissue_order = ["Ductal", "Stroma", "Fibroblast", "TLS", "Tumor", "Fatty", "Immunity infiltration", "Blank"]
tissue_color_order = ['#1f77b4', '#7f7f7f', '#ff7f0e', '#d62728', '#70CDDD', '#8c564b', '#e377c2', '#7f7f7f']
color_tissue = { k:v for k,v in zip(tissue_order, tissue_color_order) }


def combine_sc_adatas(group_id: str, df_tissue: pd.DataFrame=None) -> sc.AnnData:
    """Given name, return anndata with:
     - cell type labels
     - CAST labels
     - tissue labels

    Args:
        group_id (str): Group ID to identify the dataset.

    Returns:
        sc.Anndata: anndata object containing the combined single-cell data.
    """
    file_h5ad = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/v1/11.single_cell/{group_id}/single_cell_exp.h5ad"
    file_cast = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/v1/CAST/{group_id}/single_cell.h5ad"
    adata_sc = sc.read_h5ad(file_h5ad)
    adata_cast = sc.read_h5ad(file_cast)
    adata_sc.obs["CAST_label"] = pd.Categorical(adata_cast.obs["CAST_label"].astype(str).tolist())
    adata_sc.uns["cell_type_colors"] = [leaf_color_map[cell_type] for cell_type in adata_sc.obs["cell_type"].cat.categories]
    if df_tissue is None:
        adata_all = sc.read_h5ad("/cluster/home/bqhu_jh/projects/panlab/code/wujiangchao/ALLTLS/merged_adata_anno.h5ad")
        df_tissue = adata_all[adata_all.obs["sample"] == "ALL-TLS_100"].obs[["y", "x", "tissue"]]

    __assign_tissue_to_cells(adata_sc, df_tissue, tile_size=128)
    adata_sc.obs["tissue"] = pd.Categorical(
        adata_sc.obs["tissue"].astype(str).tolist(),
        categories=tissue_order,
        ordered=True
    )
    adata_sc.uns["tissue_colors"] = [color_tissue[tissue] for tissue in adata_sc.obs["tissue"].cat.categories]
    return adata_sc

def plot_stacked(adata_sc: sc.AnnData, group_name="tissue", frac_column="cell_type", value_column="CAST_label", ax=None) -> plt.figure:
    """
    Plot a stacked bar chart showing the cell type composition in each tissue.

    Args:
        adata_sc (sc.AnnData): anndata object containing single-cell data with tissue and cell type information.

    Returns:
        _type_: _description_
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        print("No axis provided, creating a new figure and axis.")

    df_ratio = adata_sc.obs.pivot_table(
        index=group_name,
        columns=frac_column,
        values=value_column,
        aggfunc=len
    ).fillna(0).astype(int).sort_index(axis=1)
    df_ratio_pct = df_ratio.div(df_ratio.sum(axis=1), axis=0) * 100

    # Create a custom colormap from leaf_color_map
    cell_types_in_data = df_ratio_pct.columns.tolist()
    colors_for_data = [leaf_color_map[ct] for ct in cell_types_in_data]
    custom_cmap = ListedColormap(colors_for_data)

    # Use the custom colormap
    df_ratio_pct.plot(
        kind="bar",
        stacked=True,
        colormap=custom_cmap,
        ax=ax
    )
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Tissue")
    ax.set_title("Cell type composition in each tissue (stacked bar)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if fig is None:
        plt.close()

    return df_ratio_pct, fig

def plot_cell_type(
        adata_sc: sc.AnnData, 
        group_name: str="cell_type",
        size: int=1,
        alpha: float=0.3,
        crop_region: Tuple[int, int, int, int]=None,
        da_HE: np.ndarray|da.Array=None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.figure:
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    x_min, x_max = 0, 16000
    y_min, y_max = 0, 16000
    if adata_sc is not None:
        if crop_region is not None:
            x_min, x_max, y_min, y_max = crop_region
            mask = (
                (adata_sc.obs["x"] >= x_min) & (adata_sc.obs["x"] <= x_max) &
                (adata_sc.obs["y"] >= y_min) & (adata_sc.obs["y"] <= y_max)
            )
            adata_sc = adata_sc[mask]

        df_cells = adata_sc.obs[["x", "y", group_name]].copy()
        if f"{group_name}_colors" in adata_sc.uns:
            color_cells = {k: v for k, v in zip(adata_sc.obs[group_name].cat.categories, adata_sc.uns[f"{group_name}_colors"])}
            ax.scatter(df_cells["x"]-x_min, df_cells["y"]-y_min, c=df_cells[group_name].map(color_cells), s=size, alpha=alpha, edgecolor='none')
        elif isinstance(adata_sc.obs[group_name].dtype, pd.CategoricalDtype):
            cmap = matplotlib.cm.get_cmap("tab20")
            categories = adata_sc.obs[group_name].cat.categories
            color_cells = {k: matplotlib.colors.rgb2hex(cmap(i % 20)) for i, k in enumerate(categories)}
            ax.scatter(df_cells["x"]-x_min, df_cells["y"]-y_min, c=df_cells[group_name].map(color_cells), s=size, alpha=alpha, edgecolor='none')
        else:
            ax.scatter(df_cells["x"]-x_min, df_cells["y"]-y_min, c=df_cells[group_name], s=size, alpha=alpha, edgecolor='none')
        

    if da_HE is not None:
        ax.imshow(da_HE[y_min:y_max, x_min:x_max, :].compute())

    ax.set_xticks(np.linspace(0, x_max-x_min, 5))
    ax.set_yticks(np.linspace(0, y_max-y_min, 5))

    ax.set_xticklabels(np.linspace(x_min, x_max, 5).astype(int))
    ax.set_yticklabels(np.linspace(y_min, y_max, 5).astype(int))
    ax.set_aspect('equal', adjustable='box')
    if da_HE is None:
        ax.invert_yaxis()

    plt.close()
    return fig

def plot_spatial(adata_sc: sc.AnnData, group_name: str=None, spot_size=32, crop_region: Tuple[int, int, int, int]=None) -> plt.figure:
    """Plot spatial data from an AnnData object.
    
    Args:
        adata_sc (sc.AnnData): AnnData object containing spatial data.
        group_name (str, optional): Name of the group to color by. If None,
            each variable will be plotted separately. Defaults to None.
        spot_size (int, optional): Size of the spots in the plot. Defaults to 32.

    Returns:
        matplotlib.figure.Figure: Figure object containing the spatial plots.
    """
    fig = plt.figure(figsize=(15, 15))
    
    if group_name is not None:
        for i, name in enumerate(sorted(adata_sc.obs[group_name].unique())):
            ax = fig.add_subplot(4, 6, i+1)
            sc.pl.spatial(adata_sc[adata_sc.obs[group_name]==name], color=group_name, spot_size=spot_size, ax=ax, show=False, return_fig=False)
            ax.set_title(f"{name}")
            if crop_region is not None:
                x_beg, x_end, y_beg, y_end = crop_region
                ax.set_xlim(x_beg,  x_end)
                ax.set_ylim(y_beg,  y_end)
                ax.invert_yaxis()

    else:
        for i, name in enumerate(sorted(adata_sc.var_names)):
            ax = fig.add_subplot(4, 6, i+1)
            sc.pl.spatial(adata_sc, color=name, spot_size=spot_size, ax=ax, show=False, return_fig=False)
            ax.set_title(f"{name}")
            if crop_region is not None:
                x_beg, x_end, y_beg, y_end = crop_region
                ax.set_xlim(x_beg,  x_end)
                ax.set_ylim(y_beg,  y_end)
                ax.invert_yaxis()
    
    plt.close()
    return fig


def __assign_tissue_to_cells(adata_sc: sc.AnnData, df_tissue: pd.DataFrame, tile_size: int=128):
    x = adata_sc.obs["x"].values
    y = adata_sc.obs["y"].values
    tissue_labels = np.full(len(adata_sc), "Unknown", dtype=object)

    for _, row in df_tissue.iterrows():
        mask = (
            (x >= row["x"]) & (x < row["x"]+tile_size) &
            (y >= row["y"]) & (y < row["y"]+tile_size)
        )
        tissue_labels[mask] = row["tissue"]

    adata_sc.obs["tissue"] = tissue_labels.astype(str).tolist()
