from typing import Tuple, List, Union, Optional, Dict

from scipy.sparse import issparse
import numpy as np
import dask.array as da
import pandas as pd
import scanpy as sc
from matplotlib.colors import ListedColormap
import trimesh
from scipy.spatial import Delaunay, ConvexHull
# import omnialigner as om
import matplotlib
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors
# plt = om.pl.plt


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
        adata: sc.AnnData, 
        group_name: str="cell_type",
        size: int=1,
        alpha: float=0.3,
        crop_region: Tuple[int, int, int, int]=None,
        da_HE: np.ndarray|da.Array=None,
        HE_zoom_scaled: float=1.,
        basis: str="obs",
        cmap=None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (10, 6),
        legend: bool=True,
        return_dict: bool=False,
        save_type: str = "all",
        **kwargs
    ) -> Tuple[plt.figure, Dict[str, Union[np.ndarray, pd.DataFrame, sc.AnnData]]]:

    valid_save_types = ["all", "img", "boxes"]
    if save_type not in valid_save_types:
        raise ValueError(f"save_type must be one of {valid_save_types}, got '{save_type}'")
    
    should_draw_img = save_type in ["all", "img"]
    should_draw_boxes = save_type in ["all", "boxes"]
    
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    X_min, Y_min = 0, 0
    X_max, Y_max = 1280, 1280
    if crop_region is not None:
        Y_min, Y_max, X_min, X_max = crop_region
  
    if adata is not None:
        if basis == "obs":
            np_x, np_y = adata.obs["x"].values, adata.obs["y"].values
        else:
            np_x, np_y = adata.obsm[basis][:, 0], adata.obsm[basis][:, 1]

        mask = np.ones(len(adata), dtype=bool)
        if crop_region is not None:
            mask = (
                (np_x >= X_min) & (np_x <= X_max) &
                (np_y >= Y_min) & (np_y <= Y_max)
            )
            adata_sc = adata[mask].copy()
        else:
            adata_sc = adata
            X_max, Y_max = np_x.max(), np_y.max()

        if group_name in adata_sc.obs.columns:
            np_vals = adata_sc.obs[group_name].values
        elif group_name in adata_sc.var_names:
            np_mat = adata_sc[:, group_name].X
            if issparse(np_mat):
                np_mat = np_mat.toarray()
            np_vals = np_mat.flatten()
        else:
            raise ValueError(f"Group name '{group_name}' not found in obs or var of the AnnData object.")

        df_cells = pd.DataFrame({
            "x": np_x[mask]-X_min,
            "y": np_y[mask]-Y_min,
            group_name: np_vals
        })
        df_cells["x"] = df_cells["x"] * HE_zoom_scaled
        df_cells["y"] = df_cells["y"] * HE_zoom_scaled
        if isinstance(df_cells[group_name].dtype, str):
            df_cells[group_name] = pd.Categorical(df_cells[group_name])

        if isinstance(df_cells[group_name].dtype, pd.CategoricalDtype):
            if cmap is None:
                cmap = sc.pl.palettes.default_20

            categories = adata.obs[group_name].cat.categories
            color_cells = {k: cmap[i % len(cmap)] for i, k in enumerate(categories)}
            
            if should_draw_img:
                scatter = ax.scatter(df_cells["x"], df_cells["y"], c=df_cells[group_name].map(color_cells), 
                                     s=size, alpha=alpha, edgecolor='none')
            
            if should_draw_boxes and legend:
                legend_elements = [Patch(facecolor=color_cells[cat], label=cat) for cat in categories]
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
          
        else:
            if cmap is None:
                cmap = "viridis"

            vmin = kwargs.get("vmin", df_cells[group_name].min())
            vmax = kwargs.get("vmax", df_cells[group_name].max())
            
            if should_draw_img:
                scatter = ax.scatter(df_cells["x"], df_cells["y"], c=df_cells[group_name], 
                                     s=size, alpha=alpha, edgecolor='none', cmap=cmap, vmin=vmin, vmax=vmax)
            
            if should_draw_boxes and legend:
                try:
                    plt.colorbar(scatter, ax=ax, label=group_name)
                except:
                    pass
      
    np_HE_cropped = None
    if da_HE is not None:
        Y_min_scaled = int(Y_min * HE_zoom_scaled)
        Y_max_scaled = int(Y_max * HE_zoom_scaled)
        X_min_scaled = int(X_min * HE_zoom_scaled)
        X_max_scaled = int(X_max * HE_zoom_scaled)
        np_HE_cropped = da_HE[X_min_scaled:X_max_scaled, Y_min_scaled:Y_max_scaled, :]
        
        if should_draw_img:
            ax.imshow(np_HE_cropped)

    if should_draw_boxes:
        ax.set_yticks(np.linspace(0, X_max-X_min, 5)*HE_zoom_scaled)
        ax.set_xticks(np.linspace(0, Y_max-Y_min, 5)*HE_zoom_scaled)
        ax.set_yticklabels(np.linspace(X_min, X_max, 5).astype(int))
        ax.set_xticklabels(np.linspace(Y_min, Y_max, 5).astype(int))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{group_name}")
        if da_HE is None:
            ax.invert_yaxis()
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if da_HE is None:
            ax.invert_yaxis()

    if not return_dict:
        plt.close()
        del adata_sc
        return fig

    dict_plot = {"img_HE": np_HE_cropped, "df_cells": df_cells, "adata_sub": adata_sc}
    return fig, dict_plot

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


def adata_to_trimesh(
    adata: sc.AnnData,
    group_name: str,
    basis: str = "spatial_3d",
    min_points: int = 100,
    max_edge_length: float = 10.0,
    z_scale: float = 1.0,
) -> Optional[trimesh.Trimesh]:
    """
    从AnnData生成单个类别的Trimesh
    
    Parameters:
    -----------
    adata : sc.AnnData
        单个类别的数据
    group_name : str
        用于命名的列名(主要用于日志)
    basis : str
        3D坐标的obsm键名
    min_points : int
        最小点数阈值
    max_edge_length : float
        最大边长阈值(一刀切)
    z_scale : float
        Z轴缩放因子
        
    Returns:
    --------
    mesh : trimesh.Trimesh or None
        生成的mesh,失败则返回None
    """
    
    if len(adata) < min_points:
        print(f"Category '{group_name}': too few points ({len(adata)}), skipping")
        return None
    
    # 获取坐标
    coords = adata.obsm[basis].copy()
    # coords[:, 2] *= z_scale  # Z轴缩放
    
    try:
        # Delaunay三角化
        tri = Delaunay(coords, qhull_options='QJ')
        
        # 过滤长边(使用固定阈值)
        valid_simplices = []
        for simplex in tri.simplices:
            vertices = coords[simplex]
            edge_lengths = [
                np.linalg.norm(vertices[i] - vertices[j])
                for i in range(4) for j in range(i+1, 4)
            ]
            
            if all(length <= max_edge_length for length in edge_lengths):
                valid_simplices.append(simplex)
        
        if len(valid_simplices) == 0:
            print(f"Category '{group_name}': no valid simplices after filtering")
            return None
        
        # 创建mesh
        coords[:, 2] *= z_scale  # Z轴缩放
        mesh = trimesh.Trimesh(
            vertices=coords,
            faces=np.array(valid_simplices),
            process=False
        )
        
        filter_ratio = len(valid_simplices) / len(tri.simplices)
        print(f"Category '{group_name}': {len(coords)} points, "
              f"{len(valid_simplices)}/{len(tri.simplices)} simplices "
              f"({100*filter_ratio:.1f}%), max_edge={max_edge_length:.2f}")
        
        return mesh
        
    except Exception as e:
        print(f"Failed to create mesh for '{group_name}': {e}")
        return None


def adata_categorical_to_meshes(
    adata: sc.AnnData,
    group_name: str,
    basis: str = "spatial_3d",
    min_points: int = 100,
    z_scale: float = 1.0,
    cmap=None,
    **kwargs
) -> Dict[str, Dict[str, any]]:
    """
    从AnnData的分类列生成多个Trimesh和PointCloud
    
    Parameters:
    -----------
    adata : sc.AnnData
        输入数据
    group_name : str
        obs中的分类列名
    basis : str
        3D坐标的obsm键名
    min_points : int
        每个类别最小点数
    z_scale : float
        Z轴缩放因子
    cmap : color map
        颜色列表
    **kwargs : dict
        传递给adata_to_trimesh的其他参数(如max_edge_length)
        
    Returns:
    --------
    dict_results : dict
        {category_name: {'mesh': trimesh.Trimesh, 'pointcloud': trimesh.PointCloud, 'color': str}}
    """
    
    if group_name not in adata.obs.columns:
        raise ValueError(f"Column '{group_name}' not found in adata.obs")
    
    # 获取分类和颜色
    values = adata.obs[group_name].values
    if not isinstance(pd.Series(values).dtype, pd.CategoricalDtype):
        values = pd.Categorical(values)
    
    categories = values.categories
    
    if cmap is None:
        cmap = sc.pl.palettes.default_20
    
    color_map = {cat: cmap[i % len(cmap)] for i, cat in enumerate(categories)}
    
    dict_results = {}
    
    for cat in tqdm(categories, desc="Building meshes and point clouds"):
        cat_mask = (values == cat)
        adata_sub = adata[cat_mask].copy()
        
        if len(adata_sub) < min_points:
            print(f"Category '{cat}': too few points ({len(adata_sub)}), skipping")
            continue
        
        # 生成mesh
        mesh = adata_to_trimesh(
            adata_sub,
            group_name=str(cat),
            basis=basis,
            min_points=min_points,
            z_scale=z_scale,
            **kwargs
        )
        
        # 生成pointcloud
        coords = adata_sub.obsm[basis].copy()
        coords[:, 2] *= z_scale  # Z轴缩放
        point_cloud = trimesh.points.PointCloud(vertices=coords)
        
        # 设置颜色
        hex_color = color_map[cat]
        rgba = mcolors.to_rgba(hex_color)
        rgba_u8 = (np.array(rgba) * 255).astype(np.uint8)
        
        if mesh is not None:
            from trimesh.visual.material import PBRMaterial
            
            # 创建PBR材质(高亮度,无金属感)
            material = PBRMaterial(
                baseColorFactor=rgba,
                metallicFactor=0.0,
                roughnessFactor=0.8,
                emissiveFactor=[rgba[0]*0.3, rgba[1]*0.3, rgba[2]*0.3]
            )
            
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=rgba_u8
            )
            mesh.visual.material = material
        
        point_cloud.colors = np.tile(rgba_u8, (len(coords), 1))
        
        dict_results[str(cat)] = {
            'mesh': mesh,
            'pointcloud': point_cloud,
            'color': hex_color
        }

    print(f"\nSuccessfully created {len(dict_results)} object groups from {len(categories)} categories")
    return dict_results


def plot_cell_type_3d_core(
    adata: sc.AnnData,
    group_name: str,
    scene: trimesh.Scene,
    basis: str = "spatial_3d",
    cmap = None,
    show_trimesh: bool = False,
    min_points: int = 100,
    max_edge_length: float = 10.0,
    z_scale: float = 1.0,
    z_offset: float = 0.0,  # 新增参数
) -> trimesh.Scene:
    """
    核心函数: 将单个group_name的分类/连续值添加到scene中
    
    Parameters:
    -----------
    adata : sc.AnnData
        输入数据(已裁剪和缩放)
    group_name : str
        obs列名或var基因名
    scene : trimesh.Scene
        要添加对象的场景
    basis : str
        3D坐标的obsm键名
    cmap : color map
        分类用列表,连续值用matplotlib colormap
    show_trimesh : bool
        True同时生成mesh和点云,False仅点云
    min_points : int
        mesh模式下每个类别最小点数
    max_edge_length : float
        最大边长阈值(一刀切,仅mesh模式)
    z_scale : float
        Z轴缩放因子(仅mesh模式)
    z_offset : float
        Z轴偏移量,用于分离多个group
        
    Returns:
    --------
    scene : trimesh.Scene
        更新后的场景
    """
    
    # 获取值
    if group_name in adata.obs.columns:
        values = adata.obs[group_name].values
    elif group_name in adata.var_names:
        np_mat = adata[:, group_name].X
        if issparse(np_mat):
            np_mat = np_mat.toarray()
        values = np_mat.flatten()
    else:
        raise ValueError(f"Group name '{group_name}' not found")
    
    # 应用z_offset
    adata_offset = adata.copy()
    coords_offset = adata_offset.obsm[basis].copy()
    coords_offset[:, 2] += z_offset
    adata_offset.obsm[basis] = coords_offset
    
    # 分类数据
    if isinstance(values[0], (str, np.str_)) or isinstance(pd.Series(values).dtype, pd.CategoricalDtype):
        # 使用adata_categorical_to_meshes生成
        dict_results = adata_categorical_to_meshes(
            adata_offset,
            group_name=group_name,
            basis=basis,
            min_points=min_points,
            z_scale=z_scale,  # z_scale已在外部处理
            cmap=cmap,
            max_edge_length=max_edge_length
        )
        
        # 添加到scene
        for cat, data in dict_results.items():
            # 总是添加点云
            node_name = f"{group_name}_{cat}_points"
            scene.add_geometry(data['pointcloud'], node_name=node_name)
            
            # 如果show_trimesh=True且mesh存在,也添加mesh
            if show_trimesh and data['mesh'] is not None:
                mesh_name = f"{group_name}_{cat}_mesh"
                scene.add_geometry(data['mesh'], node_name=mesh_name)
        
        n_meshes = sum(1 for d in dict_results.values() if d['mesh'] is not None)
        print(f"[{group_name}] Added {len(dict_results)} point clouds and {n_meshes} meshes to scene (z_offset={z_offset:.2f})")
        
    else:
        # 连续值数据(只支持点云)
        if cmap is None:
            cmap = 'viridis'
        
        from matplotlib import cm
        from matplotlib.colors import Normalize
        
        coords_3d = adata_offset.obsm[basis]
        
        norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        colormap = cm.get_cmap(cmap)
        
        colors_rgba = colormap(norm(values))
        colors_u8 = (colors_rgba * 255).astype(np.uint8)
        
        point_cloud = trimesh.points.PointCloud(vertices=coords_3d)
        point_cloud.colors = colors_u8
        
        node_name = f"{group_name}_points"
        scene.add_geometry(point_cloud, node_name=node_name)
        
        print(f"[{group_name}] Created continuous value point cloud with {len(coords_3d)} points (z_offset={z_offset:.2f})")
    
    return scene


def plot_cell_type_3d(
    adata: sc.AnnData, 
    group_name: Union[str, List[str]] = "cell_type",
    crop_region: Optional[Tuple[float, float, float, float, float, float]] = None,
    basis: str = "spatial_3d",
    cmap: Union[None, List] = None,
    out_glb: Optional[str] = None,
    show_trimesh: bool = False,
    min_points: int = 100,
    max_edge_length: float = 10.0,
    z_scale: float = 1.0,
    z_offset_step: float = 0.2,  # 新增参数: 每个group之间的z轴间隔
) -> trimesh.Scene:
    """
    根据3D坐标和分类/连续值渲染点云或mesh场景
    
    Parameters:
    -----------
    adata : sc.AnnData
        输入数据
    group_name : str or List[str]
        obs列名或var基因名,支持多个
    crop_region : tuple, optional
        裁剪区域 [x_min, x_max, y_min, y_max, z_min, z_max]
    basis : str
        3D坐标的obsm键名
    cmap : color map or List[color map]
        分类用列表,连续值用matplotlib colormap
        如果group_name是列表,cmap也应该是对应长度的列表
    out_glb : str, optional
        输出GLB文件路径
    show_trimesh : bool
        True同时生成mesh和点云,False仅点云
    min_points : int
        mesh模式下每个类别最小点数
    max_edge_length : float
        最大边长阈值(一刀切,仅mesh模式)
    z_scale : float
        Z轴缩放因子(仅mesh模式)
    z_offset_step : float
        每个group之间的z轴间隔,避免重叠
        
    Returns:
    --------
    scene : trimesh.Scene
        可视化场景
    """
    
    # 统一转为列表
    if isinstance(group_name, str):
        group_names = [group_name]
    else:
        group_names = group_name
    
    # 处理cmap
    if cmap is None:
        cmaps = [None] * len(group_names)
    elif not isinstance(cmap, list):
        cmaps = [cmap] * len(group_names)
    else:
        cmaps = cmap
        if len(cmaps) != len(group_names):
            raise ValueError(f"cmap length ({len(cmaps)}) must match group_name length ({len(group_names)})")
    
    # 获取3D坐标
    if basis not in adata.obsm:
        raise ValueError(f"Basis '{basis}' not found in adata.obsm")
    
    coords_3d = adata.obsm[basis].copy()
    
    # 裁剪区域
    mask = np.ones(len(adata), dtype=bool)
    if crop_region is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = crop_region
        mask = (
            (coords_3d[:, 0] >= x_min) & (coords_3d[:, 0] <= x_max) &
            (coords_3d[:, 1] >= y_min) & (coords_3d[:, 1] <= y_max) &
            (coords_3d[:, 2] >= z_min) & (coords_3d[:, 2] <= z_max)
        )
        
        coords_3d = coords_3d[mask]
        coords_3d[:, 0] -= x_min
        coords_3d[:, 1] -= y_min
        coords_3d[:, 2] -= z_min
    
    # coords_3d[:, 2] *= z_scale
    adata_sub = adata[mask].copy()
    adata_sub.obsm[basis] = coords_3d
    
    # 创建场景
    scene = trimesh.Scene()
    
    # 遍历所有group_name,每个group添加不同的z_offset
    for idx, (gname, cmap_single) in enumerate(zip(group_names, cmaps)):
        print(f"\n{'='*60}")
        print(f"Processing group {idx+1}/{len(group_names)}: {gname}")
        print(f"{'='*60}")
        
        z_offset = idx * z_offset_step  # 计算当前group的z偏移
        
        scene = plot_cell_type_3d_core(
            adata=adata_sub,
            group_name=gname,
            scene=scene,
            basis=basis,
            cmap=cmap_single,
            show_trimesh=show_trimesh,
            min_points=min_points,
            max_edge_length=max_edge_length,
            z_scale=z_scale,
            z_offset=z_offset
        )
    
    # 导出GLB
    if out_glb is not None:
        scene.export(out_glb)
        print(f"\nExported to {out_glb}")
    
    return scene

