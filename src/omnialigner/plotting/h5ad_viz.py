from typing import List, Dict

import scanpy as sc
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon
import pandas as pd
import torch
import cv2
import geopandas as gpd
import igraph as ig
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Rectangle

from omnialigner.dtypes import Np_image_HWC, Np_image_Mask
from omnialigner.plotting.matplotlib_init import plt
from omnialigner.utils import tensor2im, im2tensor

def gdf_shape_to_image(
    gdf: gpd.GeoDataFrame,
    key: str = "cluster",
    color: str | list[str] | None = None,
    w: int = 1280,
    h: int = 1280,
):
    minx, miny, maxx, maxy = 0, 0, w, h
    pixel_size_x = (maxx - minx) / w
    pixel_size_y = (maxy - miny) / h

    transform = rasterio.transform.from_origin(minx, maxy, pixel_size_x, pixel_size_y)

    # ---------------------------------------------------------
    # CASE 1: color == None → 老逻辑，输出 int32 mask
    # ---------------------------------------------------------
    if color is None:
        raster = np.zeros((h, w), dtype=np.int32)
        shapes = []

        for _, row in gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, Point):
                geom = geom.buffer(row.radius)
            shapes.append((geom, int(row[key])))

        out = rasterize(
            shapes,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="int32",
        )

        return out[::-1, :]
    
    # ---------------------------------------------------------
    # CASE 2: color != None → 渲染 RGB 彩色图
    # ---------------------------------------------------------
    
    import matplotlib.colors as mcolors
    from collections import defaultdict
    # 颜色列表（若 color 是单色，则所有形状同一种颜色）
    if isinstance(color, str):
        colors = [color] * len(gdf)
    else:
        colors = color
        assert len(colors) == len(gdf), "color 列表必须和 gdf 行数一致"

    # 转为 RGB (0–255)
    def to_rgb255(c):
        rgb = mcolors.to_rgb(c)  # 0-1 float
        return tuple(int(v * 255) for v in rgb)

    colors = [to_rgb255(c) for c in colors]

    # 创建一个按颜色分组的字典
    color_groups = defaultdict(list)
    for idx, (row, col) in enumerate(zip(gdf.iterrows(), colors)):
        geom = row[1].geometry
        # 如果是 Point,应用 radius
        if isinstance(geom, Point):
            geom = geom.buffer(row[1].radius)
        color_groups[col].append((geom, idx))

    # 初始化空的 RGB 图像
    raster = np.zeros((h, w, 3), dtype=np.uint8)

    # 对每一组颜色进行 rasterize
    for color, items in color_groups.items():
        geometries = [item[0] for item in items]  # 获取所有几何体
        indices = [item[1] for item in items]    # 获取对应索引
        
        # 对该颜色的所有几何体进行一次性 rasterize
        mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )[::-1, :]  # 翻转方向

        # 填充颜色
        for i in range(3):
            raster[:, :, i][mask == 1] = color[i]

    return raster


def keypoints_gpd(np_coords: np.ndarray, key: Dict=None, color_key: Dict=None, radius: int=2):
    points = [Point(x, y) for x, y in zip(np_coords[:,0], np_coords[:,1])]
    gdf = gpd.GeoDataFrame(geometry=points)
    if key is None:
        key = {"clusters": 1}
    
    for k, v in key.items():
        gdf[k] = v

    if color_key is not None:
        for k, v in color_key.items():
            gdf[k] = v

    gdf["radius"] = radius
    return gdf


def adata_to_gpd(adata: sc.AnnData, key: str="clusters", radius: int=2, basis: str="spatial"):
    pd_coords = adata.obsm[basis]
    dict_keys = {}
    try:
        dict_keys[key] = adata.obs[key].values.astype(np.int32)+1
    except:
        dict_keys[key] = adata.obs[key].cat.codes.astype(np.int32).values+1
        
    gdf = keypoints_gpd(pd_coords, key=dict_keys, radius=radius)
    return gdf


def transfer_h5ad_obs_col(adata_target:sc.AnnData, adata_source:sc.AnnData, basis:str="spatial", col_in_source:str='niche', k:int=1, threshold:float=-1):
    from scipy.spatial import KDTree, distance_matrix
    he_points = adata_target.obsm[basis]
    exp_points = adata_source.obsm[basis]

    exp_tree = KDTree(exp_points)
    distances, indices = exp_tree.query(he_points, k=k)
    adata_target.obs[f'matched_{col_in_source}'] = adata_source.obs[col_in_source].iloc[indices].values

    if threshold < 0:
        nearest_dists, _ = exp_tree.query(exp_points, k=2)
        nearest_dists = nearest_dists[:, 1]
        mean_neighbor_dist = np.nanmean(nearest_dists)
        threshold = mean_neighbor_dist * 1.5


    valid_mask = distances <= threshold
    adata_target.obs[f'matched_{col_in_source}'] = adata_target.obs[f'matched_{col_in_source}'].where(valid_mask, np.nan)

    filtered_count = (~valid_mask).sum()
    print(f'Filtered {filtered_count} spots ({filtered_count/len(adata_target)*100:.1f}%) as outliers using {threshold}')
    if f"{col_in_source}_colors" in adata_source.uns:
        adata_target.uns[f"matched_{col_in_source}_colors"] = adata_source.uns[f"{col_in_source}_colors"].copy()

    return adata_target


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



    
def make_rgba_lists(image_3d_tensor, threshold=5.0):
    l_rgba_lists = []
    for i_layer in range(len(image_3d_tensor)):
        img = tensor2im(image_3d_tensor[i_layer: i_layer+1])
        img_masked = (img.mean(2) <= threshold)
        img_rgba = np.zeros([img.shape[0], img.shape[1], 4], dtype=np.uint8)
        img_rgba[:, :, 0:3] = img
        img_rgba[:, :, 3] =  255-255*img_masked
        l_rgba_lists.append(img_rgba)

    return l_rgba_lists


def crop_sub_idxs(img_test, crop_h=600, crop_w=600):
    img_test[crop_h:, crop_w:, 3] = 0
    return img_test

def tensor2img(tensor, i_layer:int=0, extend: int=1, shown: str="XY", img_size=None):
    if img_size is None:
        img_size = (256, 256)

    i_beg = max(0, i_layer)
    
    if shown=="YZ":
        i_end = min(i_layer+extend, tensor.shape[2])
        img = tensor[:, :, i_beg:i_end, :].permute(2, 1, 0, 3)
        

    elif shown=="XZ":
        i_end = min(i_layer+extend, tensor.shape[3])
        img = tensor[:, :, :, i_beg:i_end].permute(3, 1, 0, 2)
        
    else:
        i_end = min(i_layer+extend, tensor.shape[0])
        img = tensor[i_beg:i_end]

    img = torch.mean(img, axis=0).unsqueeze(0)
    img = cv2.resize(tensor2im(img), img_size)
    img[img<3] = 255
    return img


def inspect_layers(image_3d_tensor, image_3d_nonrigid=None, l_layers=None, shown="XY", extend=6, n_labels=6, img_size=None):
    n, c, h, w = image_3d_tensor.shape
    if img_size is None:
        img_size = (256, 256)
    
    if l_layers is None:
        l_layers = list(range(n))

    N = len(l_layers)
    rx, ry, rz = h/img_size[0], w/img_size[1], image_3d_tensor.shape[0]/img_size[0]
    ratios = {"X": rx, "Y": ry, "Z": rz}
    fig = plt.figure(figsize=(5, N*3))
    for idx,i_layer in enumerate(l_layers):
        ax1 = fig.add_subplot(N,2,2*idx+1)
        ax2 = fig.add_subplot(N,2,2*idx+2)
        img_slice = tensor2img(image_3d_tensor, i_layer, extend=extend, shown=shown)
        ax1.imshow(img_slice)
        label = list( set(("X", "Y", "Z")) - set(list(shown)) )[0]
        ax1.set_title(f"{label} = {i_layer}")
        ax1.set_xlabel(f'{shown[0]}')
        ax1.set_ylabel(f'{shown[1]}')
        np_xticks = np.linspace(0, 256, n_labels).astype(np.int32)
        np_xticks_label = (np_xticks*ratios[shown[0]]).astype(np.int32)
        ax1.set_xticks(np_xticks)
        ax1.set_xticklabels(np_xticks_label)
        np_yticks = np.linspace(0, 256, n_labels).astype(np.int32)
        np_yticks_label = (np_yticks*ratios[shown[1]]).astype(np.int32)
        ax1.set_yticks(np_yticks)
        ax1.set_yticklabels(np_yticks_label)
        if image_3d_nonrigid is None:
            continue
    
        img_slice = tensor2img(image_3d_nonrigid, i_layer, extend=extend, shown=shown)
        ax2.imshow(img_slice)
        ax2.set_xlabel(f'{shown[0]}')
        ax2.set_ylabel(f'{shown[1]}')
        np_xticks = np.linspace(0, 256, n_labels).astype(np.int32)
        np_xticks_label = (np_xticks*ratios[shown[0]]).astype(np.int32)
        ax2.set_xticks(np_xticks)
        ax2.set_xticklabels(np_xticks_label)
        np_yticks = np.linspace(0, 256, n_labels).astype(np.int32)
        np_yticks_label = (np_yticks*ratios[shown[1]]).astype(np.int32)
        ax2.set_yticks(np_yticks)
        ax2.set_yticklabels(np_yticks_label)

    return fig


def visualize_zstack_rgba(zstack: List[np.ndarray], spacing: tuple = (1.0, 1.0, 1.0)):
    """
    正确可视化3D RGBA z-stack数据，保持原始RGB颜色，并过滤透明区域
    """
    stack_array = np.stack(zstack[::-1], axis=0)  # 形状: (Z, H, W, C)
    # stack_array = stack_array[::-1, :, :, :][:, ::-1, :, :]
    grid = pv.ImageData()
    
    grid.dimensions = np.array([stack_array.shape[2] + 1,  # W + 1
                               stack_array.shape[1] + 1,  # H + 1  
                               stack_array.shape[0] + 1]) # Z + 1
    
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    
    rgb_data = stack_array[..., :3].astype(np.float32) / 255.0
    alpha_data = stack_array[..., 3].astype(np.float32) / 255.0
    expected_cells = stack_array.shape[0] * stack_array.shape[1] * stack_array.shape[2]
    
    if grid.n_cells == expected_cells:
        grid.cell_data["RGB"] = rgb_data.reshape(-1, 3)
        
        intensity = np.mean(rgb_data, axis=3)
        grid.cell_data["intensity"] = intensity.ravel(order="C")
        
        grid.cell_data["alpha"] = alpha_data.ravel(order="C")
    else:
        print(f"警告: 单元数不匹配! 期望 {expected_cells}, 实际 {grid.n_cells}")
    
    return grid, rgb_data, alpha_data


def crop_sub_idxs(img_test, x_start, y_start, crop_h=600, crop_w=600):
    """修改指定区域外的透明度"""
    # 创建副本以避免修改原始数据
    img_copy = img_test.copy()
    # 创建一个全透明的掩码
    mask = np.ones((img_copy.shape[0], img_copy.shape[1]), dtype=bool)
    # 设置裁剪区域为不透明
    mask[y_start:y_start+crop_h, x_start:x_start+crop_w] = False
    # 将裁剪区域外的alpha设置为0
    img_copy[mask, 3] = 0
    return img_copy

def create_3d_browser(np_3d, initial_z=0, initial_x=340, initial_y=340):
    """创建3D数据浏览器"""
    
    # 获取数据维度
    depth, height, width, _ = np_3d.shape
    crop_h, crop_w = 600, 600
    
    # 创建图形和子图
    fig, (ax_main, ax_xy, ax_xz, ax_yz) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3D Volume Browser', fontsize=16, fontweight='bold')
    
    # 初始位置
    z_pos = initial_z
    x_pos = initial_x
    y_pos = initial_y
    
    # 主视图：XY平面（在指定Z深度）
    rect_main = Rectangle((x_pos, y_pos), crop_w, crop_h, 
                         linewidth=2, edgecolor='red', facecolor='none')
    # ax_main.add_patch(rect_main)
    
    # XY投影视图（沿Z轴投影）
    xy_proj = np.mean(np_3d[:, :, :, :3], axis=0).astype(np.uint8)
    im_xy = ax_xy.imshow(xy_proj, aspect='auto', cmap='gray')
    ax_xy.set_title('XY Projection (Along Z)')
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    
    # 在XY投影上添加当前位置指示器
    line_xy_x = ax_xy.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7)
    line_xy_y = ax_xy.axhline(y=y_pos, color='red', linestyle='--', alpha=0.7)
    point_xy = ax_xy.scatter([x_pos], [y_pos], c='red', s=50, marker='o', 
                           edgecolors='white')
    
    # XZ投影视图（沿Y轴投影）
    xz_proj = np.mean(np_3d[:, :, :, :3], axis=1).astype(np.uint8)
    im_xz = ax_xz.imshow(xz_proj, aspect='auto', cmap='gray')
    ax_xz.set_title('XZ Projection (Along Y)')
    ax_xz.set_xlabel('X')
    ax_xz.set_ylabel('Z')
    
    # 在XZ投影上添加当前位置指示器
    line_xz_x = ax_xz.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7)
    line_xz_z = ax_xz.axhline(y=z_pos, color='red', linestyle='--', alpha=0.7)
    point_xz = ax_xz.scatter([x_pos], [z_pos], c='red', s=50, marker='o', 
                           edgecolors='white')
    
    # YZ投影视图（沿X轴投影）
    yz_proj = np.mean(np_3d[:, :, :, :3], axis=2).astype(np.uint8)
    im_yz = ax_yz.imshow(yz_proj, aspect='auto', cmap='gray')
    ax_yz.set_title('YZ Projection (Along X)')
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    
    # 在YZ投影上添加当前位置指示器
    line_yz_y = ax_yz.axvline(x=y_pos, color='red', linestyle='--', alpha=0.7)
    line_yz_z = ax_yz.axhline(y=z_pos, color='red', linestyle='--', alpha=0.7)
    point_yz = ax_yz.scatter([y_pos], [z_pos], c='red', s=50, marker='o', 
                           edgecolors='white')
    
    plt.tight_layout()
    
    def update_positions():
        """更新所有视图的位置指示器"""
        # 更新主视图
        img_main = np_3d[z_pos]
        # cropped_main = crop_sub_idxs(img_main, x_pos, y_pos, crop_h, crop_w)
        # im_main.set_data(cropped_main)
        # ax_main.set_title(f'XY Plane at Z={z_pos}')
        
        # 更新主视图的矩形位置
        rect_main.set_xy((x_pos, y_pos))
        
        # 更新XY投影的指示器
        line_xy_x.set_xdata([x_pos, x_pos])
        line_xy_y.set_ydata([y_pos, y_pos])
        point_xy.set_offsets([[x_pos, y_pos]])
        
        # 更新XZ投影的指示器
        line_xz_x.set_xdata([x_pos, x_pos])
        line_xz_z.set_ydata([z_pos, z_pos])
        point_xz.set_offsets([[x_pos, z_pos]])
        
        # 更新YZ投影的指示器
        line_yz_y.set_xdata([y_pos, y_pos])
        line_yz_z.set_ydata([z_pos, z_pos])
        point_yz.set_offsets([[y_pos, z_pos]])
        
        fig.canvas.draw_idle()
    
    # 创建交互式控件（使用ipywidgets）
    z_slider = widgets.IntSlider(
        value=initial_z,
        min=0,
        max=depth-1,
        step=1,
        description='Z Position:',
        continuous_update=True
    )
    
    x_slider = widgets.IntSlider(
        value=initial_x,
        min=0,
        max=width,
        step=1,
        description='X Position:',
        continuous_update=True
    )
    
    y_slider = widgets.IntSlider(
        value=initial_y,
        min=0,
        max=height,
        step=1,
        description='Y Position:',
        continuous_update=True
    )
    
    # 定义更新函数
    def update_z(change):
        nonlocal z_pos
        z_pos = change['new']
        update_positions()
    
    def update_x(change):
        nonlocal x_pos
        x_pos = change['new']
        update_positions()
    
    def update_y(change):
        nonlocal y_pos
        y_pos = change['new']
        update_positions()
    
    # 连接事件
    z_slider.observe(update_z, names='value')
    x_slider.observe(update_x, names='value')
    y_slider.observe(update_y, names='value')
    
    # 创建控制面板
    controls = widgets.VBox([
        z_slider,
        x_slider,
        y_slider
    ])
    
    return controls, fig

# 使用ipywidgets创建更美观的界面
def create_interactive_3d_browser(l_np_3ds, z_range=5, titles: List[str]=None):
    """创建带有控制面板的交互式3D浏览器"""
    
    depth, height, width, _ = l_np_3ds[0].shape
    
    # 创建输出区域
    out = widgets.Output()
    
    # 创建滑块
    z_slider = widgets.IntSlider(
        value=depth//2,
        min=0,
        max=depth-1,
        step=1,
        description='Z Depth:',
        style={'description_width': 'initial'}
    )
    
    x_slider = widgets.IntSlider(
        value=height//2,
        min=0,
        max=width,
        step=1,
        description='X Position:',
        style={'description_width': 'initial'}
    )
    
    y_slider = widgets.IntSlider(
        value=width//2,
        min=0,
        max=height,
        step=1,
        description='Y Position:',
        style={'description_width': 'initial'}
    )
    
    # 创建重置按钮
    reset_button = widgets.Button(
        description='Reset View',
        button_style='info',
        tooltip='Reset to initial position'
    )
    
    # 创建裁剪尺寸控制
    crop_size_slider = widgets.IntSlider(
        value=600,
        min=100,
        max=800,
        step=50,
        description='Crop Size:',
        style={'description_width': 'initial'}
    )
    
    # 初始化变量
    crop_size = [600]  # 使用列表以便在闭包中修改
    
    @out.capture(clear_output=True)
    def update_display(l_np_3ds, z, x, y, crop_size_val, z_range=5):
        """更新显示"""
        
        n_objects = len(l_np_3ds)
        
        for i_obj, np_3d in enumerate(l_np_3ds):
            # XY投影（沿Z轴）
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            xy_proj = np.mean(np_3d[max(0, z):min(depth, z+z_range), :, :, :], axis=0).astype(np.uint8)
            axes[0].imshow(xy_proj, cmap='gray')
            title = f'XY Projection (Z +{z_range})'
            if titles is not None:
                title = titles[i_obj] + ' - ' + title
            axes[0].set_title(title)
            axes[0].axvline(x=x, color='red', linestyle='--', alpha=0.7)
            axes[0].axhline(y=y, color='red', linestyle='--', alpha=0.7)
            axes[0].scatter([x], [y], c='red', s=100, marker='x')
            
            # XZ投影（沿Y轴）
            xz_proj = np.mean(np_3d[:, max(0, y):min(height, y+z_range), :, :], axis=1).astype(np.uint8)
            axes[1].imshow(xz_proj, cmap='gray', aspect='auto')
            title = f'XZ Projection (Y +{z_range})'
            if titles is not None:
                title = titles[i_obj] + ' - ' + title
            axes[1].set_title(title)
            axes[1].axvline(x=x, color='red', linestyle='--', alpha=0.7)
            axes[1].axhline(y=z, color='red', linestyle='--', alpha=0.7)
            axes[1].scatter([x], [z], c='red', s=100, marker='x')
            
            # YZ投影（沿X轴）
            yz_proj = np.mean(np_3d[:, :, max(0, x):min(width, x+z_range), :], axis=2).astype(np.uint8)
            axes[2].imshow(yz_proj, cmap='gray', aspect='auto')
            title = f'YZ Projection (X +{z_range})'
            if titles is not None:
                title = titles[i_obj] + ' - ' + title
            axes[2].set_title(title)
            axes[2].axvline(x=y, color='red', linestyle='--', alpha=0.7)
            axes[2].axhline(y=z, color='red', linestyle='--', alpha=0.7)
            axes[2].scatter([y], [z], c='red', s=100, marker='x')
        
            plt.tight_layout()
            plt.show()
    
    def reset_view(b):
        """重置视图"""
        z_slider.value = depth//2
        x_slider.value = width//2
        y_slider.value = height//2
        crop_size_slider.value = 600
    
    # 连接事件
    def on_value_change(change):
        update_display(l_np_3ds, z_slider.value, x_slider.value, y_slider.value, crop_size_slider.value, z_range=z_range)
    
    z_slider.observe(on_value_change, names='value')
    x_slider.observe(on_value_change, names='value')
    y_slider.observe(on_value_change, names='value')
    crop_size_slider.observe(on_value_change, names='value')
    reset_button.on_click(reset_view)
    
    # 创建控制面板
    controls = widgets.VBox([
        widgets.HBox([z_slider, x_slider]),
        widgets.HBox([y_slider, crop_size_slider]),
        reset_button
    ])
    
    # 初始显示
    update_display(l_np_3ds, z_slider.value, x_slider.value, y_slider.value, crop_size_slider.value, z_range=z_range)
    
    # 返回完整的界面
    return widgets.VBox([controls, out])



def make_confusion_mat(
    df_tables: pd.DataFrame,
    col_row: str,
    col_col: str,
    dropna: bool = False,
    na_label: str = "<NA>",
    reorder: bool = True,
    method: str = "hungarian",  # "hungarian" or "greedy" (fallback)
    scale: str | None = None,   # None / "row" / "col" / "both"
    return_maps: bool = False,
):
    """
    Build a confusion matrix from df_tables[col_row] (rows) vs df_tables[col_col] (cols),
    then (optionally) reorder rows/cols to maximize diagonal mass.

    Args:
        df_tables : pd.DataFrame
            Table containing the two label columns.
        col_row : str
            Label column to be used as matrix rows (often ground-truth).
        col_col : str
            Label column to be used as matrix cols (often prediction).
        dropna : bool
            If False, NaNs will be treated as a category (na_label).
        reorder : bool
            Whether to reorder rows/cols to make the matrix close to diagonal.
        method : str
            "hungarian" (optimal) or "greedy" (approx).
        scale : str | None
            None, "row", "col", or "both" (row then col).
        return_maps : bool
            If True, also return the aligned row<->col pairing (dicts).

    Returns:
        cm : pd.DataFrame
            Confusion matrix (possibly reordered/scaled).
        (optional) row_to_col, col_to_row : dict
            Pairing maps produced by assignment (only for paired labels; dummies excluded).
    """

    df = df_tables[[col_row, col_col]].copy()

    if not dropna:
        df[col_row] = df[col_row].astype("object").where(df[col_row].notna(), na_label)
        df[col_col] = df[col_col].astype("object").where(df[col_col].notna(), na_label)
    else:
        df = df.dropna(subset=[col_row, col_col])

    # Raw confusion matrix (counts)
    cm = pd.crosstab(df[col_row], df[col_col], dropna=False)

    row_labels = list(cm.index)
    col_labels = list(cm.columns)
    A = cm.to_numpy(dtype=float)
    r, c = A.shape

    row_to_col = {}
    col_to_row = {}

    if reorder and (r > 0) and (c > 0):
        n = max(r, c)
        Ap = np.zeros((n, n), dtype=float)
        Ap[:r, :c] = A

        assignment = None

        if method == "hungarian":
            try:
                from scipy.optimize import linear_sum_assignment
                # maximize Ap => minimize (-Ap)
                rr, cc = linear_sum_assignment(-Ap)
                assignment = (rr, cc)
            except Exception:
                method = "greedy"  # fallback

        if method == "greedy":
            # Greedy matching: repeatedly pick the largest remaining cell
            used_r, used_c = set(), set()
            pairs = []
            # flatten candidates only from real block (r x c)
            coords = [(i, j, Ap[i, j]) for i in range(r) for j in range(c)]
            coords.sort(key=lambda x: x[2], reverse=True)
            for i, j, v in coords:
                if i in used_r or j in used_c:
                    continue
                used_r.add(i); used_c.add(j)
                pairs.append((i, j))
            # add dummy matches for completeness (not strictly needed)
            rr = np.array([p[0] for p in pairs], dtype=int)
            cc = np.array([p[1] for p in pairs], dtype=int)
            assignment = (rr, cc)

        rr, cc = assignment

        # Keep only real pairings (row<r and col<c)
        pairs = []
        for i, j in zip(rr, cc):
            if i < r and j < c:
                pairs.append((i, j, A[i, j]))

        # Sort matched pairs by strength so high-mass pairs sit early on diagonal
        pairs.sort(key=lambda x: x[2], reverse=True)

        paired_rows = [i for i, _, _ in pairs]
        paired_cols = [j for _, j, _ in pairs]

        # Append unpaired rows/cols at the end (preserve original order)
        remaining_rows = [i for i in range(r) if i not in set(paired_rows)]
        remaining_cols = [j for j in range(c) if j not in set(paired_cols)]

        row_order = paired_rows + remaining_rows
        col_order = paired_cols + remaining_cols

        cm = cm.iloc[row_order, col_order]

        # Build pairing maps by label
        for i, j, _ in pairs:
            rl = row_labels[i]
            cl = col_labels[j]
            row_to_col[rl] = cl
            col_to_row[cl] = rl

    # Scaling
    if scale is not None:
        s = scale.lower()
        cm_scaled = cm.astype(float)

        def _safe_div(df_num: pd.DataFrame, denom: np.ndarray, axis: int):
            denom = denom.copy().astype(float)
            denom[denom == 0] = 1.0
            if axis == 1:  # row
                return df_num.div(denom, axis=0)
            else:          # col
                return df_num.div(denom, axis=1)

        if s == "row":
            cm_scaled = _safe_div(cm_scaled, cm_scaled.sum(axis=1).to_numpy(), axis=1)
        elif s == "col":
            cm_scaled = _safe_div(cm_scaled, cm_scaled.sum(axis=0).to_numpy(), axis=0)
        elif s == "both":
            cm_scaled = _safe_div(cm_scaled, cm_scaled.sum(axis=1).to_numpy(), axis=1)
            cm_scaled = _safe_div(cm_scaled, cm_scaled.sum(axis=0).to_numpy(), axis=0)
        else:
            raise ValueError("scale must be None/'row'/'col'/'both'")

        cm = cm_scaled

    if return_maps:
        return cm, row_to_col, col_to_row
    return cm



def adata_to_png(adata, key="leiden", w=1280, h=1280, basis="spatial_scaled", radius=2):
    pd_coords = adata.obsm[basis]
    dict_keys = {}
    try:
        dict_keys[key] = adata.obs[key].values.astype(np.int32)+1
    except:
        dict_keys[key] = adata.obs[key].cat.codes.astype(np.int32).values+1
        
    gdf = keypoints_gpd(pd_coords, key=dict_keys, radius=radius)
    np_uint8 = gdf_shape_to_image(gdf, key=key, w=w, h=h)[:, :, np.newaxis]
    return np_uint8

def calculate_h5ad_column_metrics(
        adata0: sc.AnnData,
        adata1: sc.AnnData,
        key0: str="cluster",
        key1: str="niche",
        basis: str="spatial",
        w:int=1280,
        h:int=1280,
        radius0: int=4,
        radius1: int=48,
        metric_funcs=None
    ):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    np_img0 = adata_to_png(adata0, key=key0, w=w, h=h, basis=basis, radius=radius0)
    np_img1 = adata_to_png(adata1, key=key1, w=w, h=h, basis=basis, radius=radius1)
    if metric_funcs is None:
        metric_funcs = {
            "nmi": normalized_mutual_info_score,
            "ari": adjusted_rand_score
        }

    dict_result = {}
    for metric_name, metric_func in metric_funcs.items():
        np_x = np.ravel(np_img0).astype(np.uint8)
        np_y = np.ravel(np_img1).astype(np.uint8)
        idx = (np_x>0) * (np_y>0)
        dict_result[metric_name] = metric_func(np_x[idx], np_y[idx])

    return dict_result



def generate_color_dict(adata: sc.AnnData, tag: str, cmap: List[str]=None):
    if cmap is None:
        cmap = sc.pl.palettes.default_102

    l_name_used = [ v for v in adata.obs[tag].value_counts().index ]
    adata.obs["cluster"] = [ int(l_name_used.index(v)) if v in l_name_used else -1 for v in adata.obs[tag] ]
    color_cells = {k+1: cmap[i % len(cmap)] for i, k in enumerate(range(len(l_name_used)))}
    color_cells[-1] = "#888888"
    color_dict = {"l_name_used": l_name_used, "color_cells": color_cells}
    return color_dict


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-255)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def tensor3d_cat_to_rgb(tensor, color_cells):
    """
    Converting a 3D tensor of category labels (N, 1, H, W) to an RGB image tensor (N, 3, H, W) using a color mapping.

    Args:
        tensor: PyTorch tensor of shape (N, 1, H, W) containing category labels (integers).
        color_cells: dict mapping category labels (integers) to hex color strings (e.g., {0: "#FF0000", 1: "#00FF00", ...}).
    
    Returns:
        tensor_rgb: PyTorch tensor of shape (N, 3, H, W)
    """
    max_category = max(color_cells.keys())
    color_map = np.zeros((max_category + 1, 3), dtype=np.uint8)
    for cat_id, hex_color in color_cells.items():
        color_map[cat_id] = hex_to_rgb(hex_color)


    N, _, H, W = tensor.shape
    tensor_rgb = torch.zeros((N, 3, H, W), dtype=torch.float32)

    for i in range(N):
        category_img = (tensor[i, 0].numpy()).astype(np.int32)
        category_img = np.clip(category_img, -1, max_category)
        rgb_img = color_map[category_img]
        tensor_rgb[i] = torch.from_numpy(rgb_img.transpose(2, 0, 1) / 255.0)

    return tensor_rgb