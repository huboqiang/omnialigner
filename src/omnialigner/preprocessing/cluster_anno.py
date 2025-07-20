import sys
import os
from typing import Dict, List, Tuple, Union
import json

import yaml
import pandas as pd
import numpy as np
import cv2
import torch
import scanpy as sc
from tqdm import tqdm
from matplotlib.patches import Rectangle

import omnialigner as om
from omnialigner.omni_3D import Omni3D
from omnialigner.plotting.matplotlib_init import plt
from omnialigner.cache_files import StageSampleTag
from omnialigner.utils.sd_zarr import load_spatial_adata_from_h5


def clustering_adata(adata, resolutions=None):
    """
    Perform clustering on the given AnnData object.
    
    Parameters:
    adata: AnnData object containing the data to be clustered.
    
    Returns:
    None
    """
    if resolutions is None:
        resolutions = [0.2, 0.4, 0.6, 0.8, 1.0]

    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    for resolution in tqdm(resolutions):
        sc.tl.leiden(adata, key_added=f"leiden_{resolution:.1f}", flavor="igraph", resolution=resolution)

    return adata

def clustering_adata_kmeans(adata, n_clusters=[10, 15, 20]):
    """
    Perform KMeans clustering on the given AnnData object.
    
    Parameters:
    adata: AnnData object containing the data to be clustered.
    n_clusters: int, number of clusters for KMeans.
    
    Returns:
    None
    """
    from sklearn.cluster import KMeans
    X = adata.X
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(X)
        adata.obs[f"kmeans_{n_cluster}"] = pd.Categorical(kmeans.labels_.astype(str))
    return adata


def fetch_img_from_coords(
    coords: pd.DataFrame,
    image: np.ndarray,
    label_col: str = "label",
    n_rand: int = 3,
    crop_size: Tuple[int, int] = (64, 64),   # (w, h)
    replace: bool = False,
    random_state: Union[int, np.random.Generator, None] = None,
) -> Dict[Union[int, str], List[Tuple[np.ndarray, Tuple[int, int]]]]:
    """
    Randomly sample coordinates per label and crop fixed-size windows from an image.

    Args:
        coords (pd.DataFrame): DataFrame with columns "x", "y", and the label column.
        image (np.ndarray): Source image array of shape (H, W[, C]).
        label_col (str): Name of the label column in `coords`. Default is "label".
        n_rand (int): Number of random samples to draw per label. Default is 3.
        crop_size (Tuple[int, int]): Width and height of the crop window (w, h). Default is (64, 64).
        replace (bool): Whether to sample with replacement. Default is False.
        random_state (Union[int, np.random.Generator, None]): Seed or RNG for reproducibility. Default is None.

    Returns:
        Dict[Union[int, str], List[Tuple[np.ndarray, Tuple[int, int]]]]:
            A dictionary mapping each label to a list of tuples, each containing:
              - cropped image patch (np.ndarray)
              - top‐left corner coordinates of the crop (x, y)

    Raises:
        ValueError: If `coords` does not contain the required columns "x", "y", and `label_col`.
    """
    required_cols = {"x", "y", label_col}
    if not required_cols.issubset(coords.columns):
        raise ValueError(f"`coords` 必须包含列 {required_cols}")

    H, W = image.shape[:2]
    w, h = crop_size
    rng = np.random.default_rng(random_state)

    dict_label_cropped: Dict[Union[int, str], List[Tuple[np.ndarray, Tuple[int, int]]]] = {}

    for label, group in coords.groupby(label_col):
        if len(group) == 0:
            continue
        size = min(len(group), n_rand) if not replace else n_rand
        idx = rng.choice(group.index, size=size, replace=replace)

        outputs: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        for i in idx:
            x, y = int(coords.at[i, "x"]), int(coords.at[i, "y"])
            # 边界安全处理
            x_clip = max(0, min(x, W - w))
            y_clip = max(0, min(y, H - h))

            # print(f"Processing label {label}, idx {i}: ({x}, {y}), clipped to ({x_clip}, {y_clip}), crop size {crop_size}")
            crop = image[y_clip : y_clip + h, x_clip : x_clip + w].copy()
            outputs.append((crop, (x_clip, y_clip)))

        dict_label_cropped[label] = outputs

    return dict_label_cropped


def viz_dict_label_cropped(
    dict_label_cropped: Dict[Union[int, str], List[Tuple[np.ndarray, Tuple[int, int]]]],
    figsize: Tuple[int, int] | None = None,
    cmap: str | None = None,
    show: bool = True,
):

    if not dict_label_cropped:
        raise ValueError("dict_label_cropped 为空，无法可视化。")

    labels = list(dict_label_cropped.keys())
    n_rows = len(labels)
    n_cols = max(len(v) for v in dict_label_cropped.values())

    figsize = figsize or (n_cols * 2.5, n_rows * 2.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r, label in enumerate(labels):
        samples = dict_label_cropped[label]
        for c in range(n_cols):
            ax = axes[r, c]
            # ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if c < len(samples):
                crop, (x, y) = samples[c]
                if crop.ndim == 2 or crop.shape[2] == 1:  # 灰度
                    ax.imshow(crop.squeeze(), cmap=cmap or "gray")
                else:
                    ax.imshow(crop)
                ax.set_title(f"({x}, {y})", fontsize=8)
            if c == 0:
                ax.set_ylabel(str(label), rotation=0, va="center", ha="right", fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def viz_boxes_on_img(
    image: np.ndarray,
    dict_label_cropped: Dict[Union[int, str], List[Tuple[np.ndarray, Tuple[int, int]]]],
    crop_size: Tuple[int, int],
    linewidth: int = 2,
    alpha: float = 1.0,
    figsize: Tuple[int, int] | None = None,
    cmap_name: str = "tab20",
    show: bool = True,
    start_idx: int = 1,
    font_size: int = 8,
    text_offset: Tuple[int, int] = (2, 8),
):
    """
    在原图上绘制所有裁剪窗口并标注序号。

    Parameters
    ----------
    image : np.ndarray
        原图 (H, W, C) uint8。
    dict_label_cropped : Dict[label, List[(crop, (x, y))]]
        `fetch_img_from_coords` 输出。
    crop_size : (w, h)
        与裁剪时相同的窗口尺寸。
    linewidth : int
        矩形边框线宽。
    alpha : float
        线条透明度。
    figsize : (w, h) | None
        figure 大小，默认按图像尺寸自动缩放。
    cmap_name : str
        不同 label 的颜色映射，默认 "tab20"。
    show : bool
        是否立即 plt.show()。
    start_idx : int
        给序号编号的起始值（默认 1）。
    font_size : int
        序号文本字号。
    text_offset : (dx, dy)
        文本相对于左上角偏移量，单位像素。
    """
    if not dict_label_cropped:
        raise ValueError("dict_label_cropped 为空，无法绘制。")

    plt = om.pl.plt
    labels = list(dict_label_cropped.keys())
    cmap = plt.get_cmap(cmap_name, max(len(labels), 1))
    label2color = {lab: cmap(i) for i, lab in enumerate(labels)}
    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    w, h = crop_size
    for label in labels:
        color = label2color[label]
        for idx, (_, (x, y)) in enumerate(dict_label_cropped[label], start=start_idx):
            x = x / 40
            y = y / 40
            h = h / 40
            w = w / 40
            # 绘制矩形框
            rect = Rectangle(
                (x, y),
                w,
                h,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                alpha=alpha,
            )
            ax.add_patch(rect)

            # 添加序号文本
            tx, ty = x + text_offset[0], y + text_offset[1]
            ax.text(
                tx,
                ty,
                f"{idx}",
                fontsize=font_size,
                color=color,
                weight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

    handles = [
        Rectangle((0, 0), 1, 1, edgecolor=label2color[l], facecolor="none", linewidth=2)
        for l in labels
    ]
    ax.legend(handles, [str(l) for l in labels], loc="upper right", frameon=False)

    if show:
        plt.show()
    return fig, ax

def extract_dict(dict_label_cropped):
    dict_out = {}
    for label, crops in dict_label_cropped.items():
        dict_out[label] = [crop[1] for crop in crops]

    return dict_outgroup_idict_outgroup_id,


def cluster_h5py(
                group_id: str,
                path_trident: str,
                 model_name: str = "conch_v15",
                 cluster_algo: str = "kmeans",
                 attention: bool = True,
                 dict_image: Dict[str, Dict[str, Union[np.ndarray, Dict[str, float]]]] = None,
                 n_kmeans_clusters: List[int] = None
    ):
    prefix = ""
    if attention:
        prefix = "attention_"
    file_h5py = f"{path_trident}/{prefix}features_{model_name}/HE_flatten.h5"
    if not os.path.exists(file_h5py):
        file_h5py = f"{path_trident}/{prefix}features_{model_name}/{group_id}_flatten.h5"
    
    file_adata = f"{path_trident}/{cluster_algo}/{prefix}{model_name}.h5ad"
    #if not os.path.exists(file_adata):
    if True:
        adata = load_spatial_adata_from_h5(file_h5py)
        file_adata_ = file_h5py + "ad"
        adata.write_h5ad(file_adata_)

        adata = sc.read_h5ad(file_adata_)
        adata.obs.columns = ["y", "x"]
        crop_size = (16, 16) if attention else (512, 512)
        if cluster_algo=="kmeans":
            adata_cluster = clustering_adata_kmeans(adata.copy(), n_clusters=n_kmeans_clusters)
            adata_cluster.uns['spatial'] = dict_image
            adata_cluster.obsm['spatial'] = adata_cluster.obs[['x', 'y']].values * 0.25
            img_key=None if dict_image is None else "raw_zoom_3"
            fig = sc.pl.spatial(adata_cluster, color=[f"{cluster_algo}_{str(x)}" for x in n_kmeans_clusters], frameon=False, spot_size=50, title=f"{cluster_algo} Clustering", img_key=img_key, ncols=3, show=False, return_fig=True)
            os.makedirs(f"{path_trident}/{cluster_algo}", exist_ok=True)
            fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}.png", dpi=600)
            sc.write(file_adata, adata_cluster)


def cluster_anno(
    om_data: Omni3D,
    resolutions: List[float] = None,
    n_kmeans_clusters: List[int] = None,
    model_name: str = "conch_v15",
    cluster_algo: str = "leiden",
    attention: bool = False,
):
    """
    Cluster annotation for Omni3D data.
    
    Parameters:
        om_data (Omni3D): The Omni3D data object.
        n_neighbors (int): Number of neighbors for KNN.
        n_clusters (int): Number of clusters to form.
        n_jobs (int): Number of jobs to run in parallel.
        use_gpu (bool): Whether to use GPU for processing.
    """
    if resolutions is None:
        resolutions = [0.2, 0.4, 0.6, 0.8, 1.0]
    if n_kmeans_clusters is None:
        n_kmeans_clusters = [10, 15, 20]
    
    file_qptiff = StageSampleTag.DATA.get_file_name(i_layer=0, projInfo=om_data.proj_info)["data"]
    da_img_raw = om.tl.read_ome_tiff(file_qptiff, i_page=0, i_level=0, l_channels=[0,1,2])
    da_img_8x = om.tl.read_ome_tiff(file_qptiff, i_page=0, i_level=3, l_channels=[0,1,2]).compute()
    da_img_40x = cv2.resize(da_img_8x, (da_img_8x.shape[1] // 5, da_img_8x.shape[0] // 5))

    # 40x thumbnail with 10x coordinate scaling
    library_id = f"slide_3"
    dict_image = {
        library_id : {
            'images': {
                'raw_zoom_3': da_img_40x,
            },
            'scalefactors': {
                'tissue_raw_zoom_3_scalef': 0.1
            }
        }
    }
    pj = om_data.proj_info
    path_trident = f"{pj.root_dir}/analysis/{pj.project}/{pj.version}/trident_processed/{pj.group}/20x_256px_0px_overlap"
    prefix = ""
    if attention:
        prefix = "attention_"

    file_h5py = f"{path_trident}/{prefix}features_{model_name}/HE_flatten.h5"
    file_adata = f"{path_trident}/{cluster_algo}/{prefix}{model_name}.h5ad"
    #if not os.path.exists(file_adata):
    if True:
        adata = load_spatial_adata_from_h5(file_h5py)
        file_adata_ = file_h5py + "ad"
        adata.write_h5ad(file_adata_)

        adata = sc.read_h5ad(file_adata_)
        adata.obs.columns = ["y", "x"]
        crop_size = (16, 16) if attention else (512, 512)
        if cluster_algo=="kmeans":
            adata_cluster = clustering_adata_kmeans(adata.copy(), n_clusters=n_kmeans_clusters)
            adata_cluster.uns['spatial'] = dict_image
            adata_cluster.obsm['spatial'] = adata_cluster.obs[['x', 'y']].values * 0.25
            fig = sc.pl.spatial(adata_cluster, color=[f"{cluster_algo}_{str(x)}" for x in n_kmeans_clusters], frameon=False, spot_size=50, title=f"{cluster_algo} Clustering", img_key="raw_zoom_3", ncols=3, show=False, return_fig=True)
            os.makedirs(f"{path_trident}/{cluster_algo}", exist_ok=True)
            fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}.png", dpi=600)
            sc.write(file_adata, adata_cluster)
            
            for res in n_kmeans_clusters:
                dict_label_cropped = fetch_img_from_coords(
                    adata_cluster.obs,
                    da_img_raw,
                    label_col=f"{cluster_algo}_{str(res)}",
                    n_rand=10,
                    random_state=42,
                    crop_size=crop_size
                )
                fig, _ = viz_dict_label_cropped(dict_label_cropped)
                fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}_res_{res}_ROIs.png")
                with open(f"{path_trident}/{cluster_algo}/{prefix}{model_name}_res_{res}_ROIs.json", 'w') as f:
                    json.dump(extract_dict(dict_label_cropped), f)

                fig, ax = viz_boxes_on_img(
                    da_img_40x,
                    dict_label_cropped,
                    crop_size=crop_size,
                    show=False,
                    start_idx=1,
                    font_size=8,
                    text_offset=(2, 8)
                )
                fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}_res_{res}_ROIs_boxes.png")


        if cluster_algo == "leiden":
            adata_cluster = clustering_adata(adata.copy(), resolutions=resolutions)
            adata_cluster.uns['spatial'] = dict_image
            adata_cluster.obsm['spatial'] = adata_cluster.obs[['x', 'y']].values * 0.25
            fig = sc.pl.spatial(adata_cluster, color=[f"{cluster_algo}_{str(x)}" for x in resolutions], frameon=False, spot_size=50, title=f"{cluster_algo} Clustering", img_key="raw_zoom_3", ncols=3, show=False, return_fig=True)
            os.makedirs(f"{path_trident}/{cluster_algo}", exist_ok=True)
            fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}.pdf")
            fig = sc.pl.umap(adata_cluster, color=[f"{cluster_algo}_{str(x)}" for x in resolutions[-1:]], frameon=False, title=f"{cluster_algo} umap Clustering",  ncols=3, show=False, return_fig=True)
            fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}.umap.png")
            sc.write(file_adata, adata_cluster)
            for res in resolutions:
                dict_label_cropped = fetch_img_from_coords(
                    adata_cluster.obs,
                    da_img_raw,
                    label_col=f"leiden_{str(res)}",
                    n_rand=10,
                    random_state=42,
                    crop_size=crop_size
                )
                fig, _ = viz_dict_label_cropped(dict_label_cropped)
                fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}_res_{res}_ROIs.png")
                with open(f"{path_trident}/{cluster_algo}/{prefix}{model_name}_res_{res}_ROIs.json", 'w') as f:
                    json.dump(extract_dict(dict_label_cropped), f)

                fig, ax = viz_boxes_on_img(
                    da_img_40x,
                    dict_label_cropped,
                    crop_size=crop_size,
                    show=False,
                    start_idx=1,
                    font_size=8,
                    text_offset=(2, 8)
                )
                fig.savefig(f"{path_trident}/{cluster_algo}/{prefix}{model_name}_res_{res}_ROIs_boxes.png")

    adata_cluster = sc.read_h5ad(file_adata)
    

def run_2d(group_id):
    with open("/cluster/home/bqhu_jh/projects/panlab/code/wujiangchao/ALLTLS/config/panlab/config_pdac.yaml", 'r') as f:
        template_string = f.read()
        config_info = yaml.load(template_string, Loader=yaml.FullLoader)

    config_info["datasets"]["group"] = f"{group_id}"
    om_data = Omni3D(config_info=config_info)
    om_data.raw_is_tiff = True
    if om_data.proj_info.get_dtype(0) == "HE":
#        cluster_anno(om_data, resolutions=[0.2, 0.4, 0.6, 0.8, 1.0], model_name="conch_v15")
#        cluster_anno(om_data, resolutions=[0.2, 0.4, 0.6, 0.8, 1.0], model_name="gigapath")

        cluster_anno(om_data, n_kmeans_clusters=[10, 15, 20], model_name="conch_v15", attention=True, cluster_algo="kmeans")
        cluster_anno(om_data, n_kmeans_clusters=[10, 15, 20], model_name="gigapath", attention=True, cluster_algo="kmeans")


def run_3d(group_id):
    path_trident = f"/cluster/home/bqhu_jh/projects/scGaussian3dGen/analysis/panlab/v6/trident_processed/{group_id}/20x_256px_0px_overlap/"
    
    for model_name in ["conch_v15", "gigapath"]:
        cluster_h5py(
                    group_id=group_id,
                    path_trident=path_trident,
                    model_name=model_name,
                    cluster_algo="kmeans",
                    attention=True,
                    dict_image=None,
                    n_kmeans_clusters=[10, 15, 20]
        )

if __name__ == "__main__":
    import matplotlib
    
    matplotlib.use("Agg")
    group_id = sys.argv[1] if len(sys.argv) > 1 else "ALL-TLS_100"
    # run_2d(group_id)
    run_3d(group_id)