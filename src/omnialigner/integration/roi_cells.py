import os
import sys

import geopandas as gpd
import pandas as pd
import numpy as np
import yaml
import scanpy as sc
import dask.array as da
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binned_statistic_2d
from scipy import ndimage
from shapely.affinity import translate

import omnialigner as om
from omnialigner.align.models.grid_2d.deeperhistreg_module import warp_tensor
from omnialigner.align.refine_tile import generate_imgs, model_denoise, refine_tile
from omnialigner.plotting.h5ad_viz import gdf_shape_to_image


plt = om.pl.plt
sns = om.pl.sns
kwargs = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True}

def make_gdf_knn(g_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add KNN distance to the GeoDataFrame.
    This function computes the KNN distance for each cell in the GeoDataFrame
    and adds it as a new column.

    Args:
        g_df (gpd.GeoDataFrame): GeoDataFrame containing cell shapes and labels.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with KNN distance added.
    """
    g_df['x'] = g_df.geometry.centroid.x
    g_df['y'] = g_df.geometry.centroid.y
    # g_df["type"] =  [ dict_old.get(name) for name in g_df["name"]]
    g_df["type"] = 0
    coords = g_df[["x", "y"]].values
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='auto').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    # 距离包括自己，取第1到第10个最近邻
    g_df["knn_dist_mean"] = np.clip(distances[:, 1:11].mean(axis=1), 0, 60)
    return g_df

def yolo_like_density_boxes(df, bin_size=100, scale=10,
                            thr_pct=95, iou_thresh=0.5, use_nmm=True):
    # 1. grid histogram
    xs, ys = df['x'] / scale, df['y'] / scale
    hist_result = binned_statistic_2d(xs, ys, None, 'count', bins=bin_size)
    hist, xe, ye = hist_result.statistic, hist_result.x_edge, hist_result.y_edge
    thr = np.percentile(hist[hist > 0], thr_pct)

    # 2. binary mask & connected components
    mask = hist >= thr
    labels, n = ndimage.label(mask)                         # 4-connected by default

    # 3. boxes + scores
    boxes, scores = [], []
    for lab in range(1, n+1):
        r, c = np.where(labels == lab)
        r0, r1, c0, c1 = r.min(), r.max()+1, c.min(), c.max()+1
        score = hist[r0:r1, c0:c1].sum()
        # back-scale to original units
        boxes.append([xe[r0]*scale, ye[c0]*scale, xe[r1]*scale, ye[c1]*scale])
        scores.append(float(score))

    # 4. NMS / NMM
    keep = []
    while boxes:
        i = int(np.argmax(scores))
        box_i, score_i = boxes.pop(i), scores.pop(i)
        keep.append((box_i[0], box_i[1], box_i[2], box_i[3], score_i))
        rem_boxes, rem_scores = [], []
        for b, s in zip(boxes, scores):
            iou = _iou(box_i, b)
            if iou < iou_thresh:
                rem_boxes.append(b); rem_scores.append(s)
            elif use_nmm:           # merge instead of suppress
                merged = _merge_boxes(box_i, b)
                box_i = merged; score_i += s
        boxes, scores = rem_boxes, rem_scores

    return pd.DataFrame(keep, columns=['x_beg', 'y_beg', 'x_end', 'y_end', 'score'])

def _iou(a, b):          # (x1,y1,x2,y2)
    xa, ya, xb, yb = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def _merge_boxes(a, b):
    return [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]



def plot_high_density_regions(
        g_df: gpd.GeoDataFrame,
        img: np.ndarray, 
        group_id="figure_name",
        scale_factor=10,
        n_boxes=10,
        threshold_percentile=95,
        bin_size=100,
        iou_thresh=0.5
    ):
    """
    Plot high-density regions on the image.
    
    Args:
        g_df : pandas.DataFrame
            DataFrame containing cell coordinates
        img : numpy.ndarray
            Image array to plot on
        scale_factor : int, default 8
            Factor to scale coordinates
        n_boxes : int, default 10
            Number of highest density boxes to highlight
        threshold_percentile : int, default 95
            Percentile threshold for high density
        bin_size : int, default 100
            Number of bins for the 2D histogram
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    axs[0].imshow(img)
    sns.histplot(
        x=g_df["x"] / scale_factor,
        y=g_df["y"] / scale_factor,
        bins=bin_size,
        cmap="Blues", 
        fill=True,
        ax=axs[0]
    )
    axs[0].set_title("Cell Density Heatmap")
    axs[1].imshow(img)
    df_res = yolo_like_density_boxes(
        g_df,
        bin_size=bin_size,
        scale=scale_factor,
        thr_pct=threshold_percentile,
        iou_thresh=iou_thresh,
        use_nmm=True
    )

    N_lines = df_res[:n_boxes].shape[0]
    for i in range(N_lines):
        box = df_res.loc[i]
        rect = plt.Rectangle(
            (box["x_beg"] / scale_factor, box["y_beg"] / scale_factor),
            (box["x_end"] - box["x_beg"]) / scale_factor,
            (box["y_end"] - box["y_beg"]) / scale_factor,
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        axs[1].add_patch(rect)
        # Add text with density value
        axs[1].text(
            box["x_beg"] / scale_factor, 
            box["y_beg"] / scale_factor - 5, 
            f'#{i+1}: {box["score"]:.0f}',
            color='white', 
            fontsize=8,
            bbox=dict(facecolor='red', alpha=0.5)
        )

    axs[1].set_title(f"Top {n_boxes} High-Density Regions")
    if not os.path.exists("./fig/cell_density"):
        fig.savefig(f"./fig/cell_density/{group_id}.png", dpi=300)
        plt.close()
        df_res.to_csv(f"./fig/cell_density/pos_{group_id}.csv", index=False)
    return df_res

def get_image(om_data: om.Omni3D, i_layer:int=0, zoom_level:int=3) -> da.Array:
    om_data.set_zoom_level(zoom_level)
    image = om.align.apply_nonrigid_HD(om_data, i_layer=i_layer)
    return image



def get_image_ihc(om_data: om.Omni3D, i_layer=0, zoom_level=1, tag="rmaf"):
    om_data.set_zoom_level(zoom_level)
    if tag is not None:
        om_data.set_tag(tag)

    image = om.align.apply_nonrigid_HD(om_data, i_layer=i_layer, tag=tag, overwrite_cache=False)
    return image



def extract_cell_feats_h5ad(
        gdf_sub: gpd.GeoDataFrame,
        tensor_crop: torch.Tensor,
        x_beg: int=0,
        y_beg: int=0
) -> sc.AnnData:
    """
    Extract cell features from the cropped tensor and the corresponding labels.
    
    Args:
        gdf_sub : GeoDataFrame
            Subset of the GeoDataFrame containing cell shapes and labels.
        tensor_crop : torch.Tensor
            Cropped tensor of shape (1, C, H, W) where C is the number of channels.

    
    Returns:
        adata : sc.AnnData
            AnnData object containing the extracted cell features and metadata.
    """
        
    _, C, H, W = tensor_crop.shape
    label_mask = gdf_shape_to_image(gdf_sub, key="clusters", w=W, h=H).astype(np.int32)

    lbl = label_mask.ravel()
    feats = tensor_crop.numpy().reshape(C, -1)

    n_labels = lbl.max() -1
    counts = np.bincount(lbl, minlength=n_labels+1)[1:]
    sums = np.vstack([np.bincount(lbl, weights=feats[c], minlength=n_labels+1)[1:] for c in range(C)])

    valid = counts > 0
    sums = sums[:, valid]
    counts = counts[valid]

    cell_feats = torch.from_numpy((sums / counts).T)
    valid_labels = np.nonzero(valid)[0] + 1
    gdf_sub = gdf_sub[gdf_sub['clusters'].isin(valid_labels)].reset_index(drop=True)

    obs = pd.DataFrame({
        'x': gdf_sub['x'].values + x_beg,
        'y': gdf_sub['y'].values + y_beg,
        "knn_dist_mean": gdf_sub["knn_dist_mean"]
    }, index=gdf_sub['clusters'].astype(str))
    var = pd.DataFrame(index=[ i for i in range(cell_feats.shape[1])])
    adata = sc.AnnData(
        X=cell_feats.numpy(),
        obs=obs,
        var=var
    )

    adata.obsm['spatial'] = obs[['x','y']].values
    return adata


def process_bbox(
                om_data: om.Omni3D,
                i_box: int,
                df_bboxes: pd.DataFrame,
):
    """
    Process bounding boxes and extract cell features.

    Args:
        om_data : Omni3D
            Omni3D object containing project information.
        i_box : int
            Index of the bounding box to process.
        df_bboxes : pandas.DataFrame
            DataFrame containing bounding box coordinates and scores.    

    """
    group_id = om_data.proj_info.group

    dict_IHC_layer = om_data.proj_info.load_IHC_info()
    l_names = list(np.array(dict_IHC_layer["genes"]["P1"]))[1:7] +\
                    list(np.array(dict_IHC_layer["genes"]["P2"]))[1:7] +\
                    list(np.array(dict_IHC_layer["genes"]["P3"]))[1:7] +\
                    list(np.array(dict_IHC_layer["genes"]["P4"]))[1:7]
                    
    l_names[-2] = "MS4A1_rep2"

    df_line = df_bboxes.loc[i_box]
    x_beg_, y_beg_ = int(df_line["x_beg"]), int(df_line["y_beg"])
    x_end_, y_end_ = int(df_line["x_end"]), int(df_line["y_end"])
    img_he = get_image(om_data, i_layer=0, zoom_level=1)
    h = y_end_ - y_beg_
    w = x_end_ - x_beg_
    y_center, x_center = (y_beg_ + y_end_) // 2, (x_beg_ + x_end_) // 2
    hw = max(h, w)
    rect_tile = [y_center - hw // 2, x_center - hw // 2, y_center + hw // 2, x_center + hw // 2]
    y_beg, x_beg, y_end, x_end = rect_tile

    file_cell = f"{om_data.proj_info.root_dir}/analysis/{om_data.proj_info.project}/{om_data.proj_info.version}/10.merge_spatialdata/{group_id}/merged_{group_id}.zarr/shapes/cells/shapes.parquet"
    g_df1 = gpd.read_parquet(file_cell)
    g_df = make_gdf_knn(g_df1)
    
    gdf_sub = g_df.cx[x_beg:x_end, y_beg:y_end]
    gdf_sub["geometry"] = gdf_sub["geometry"].apply(
            lambda geom: translate(geom, xoff=-x_beg, yoff=-y_beg)
    )
    gdf_sub["clusters"] = range(1, gdf_sub.shape[0]+1)

    print("calculating cell masks")
    mask_he = gdf_shape_to_image(gdf_sub, key="clusters", w=x_end-x_beg, h=y_end-y_beg)
    l_imgs_dapi = generate_imgs(om_data, idx_tile=0, l_tiles=[rect_tile], tag="rmaf", overwrite_cache=False)
    masks_dapi, _, _, _ = model_denoise.eval(l_imgs_dapi[1:], diameter=4, channels=[0,0])

    dir_nonrigid = f"{om_data.proj_info.root_dir}/analysis/{om_data.proj_info.project}/{om_data.proj_info.version}/06.nonrigid_tiles/"
    out_prefix = f"{dir_nonrigid}/{group_id}/roi_{i_box}"

    os.makedirs(out_prefix, exist_ok=True)

    l_cell_masks = [mask_he] + masks_dapi
    print("calculating nonrigid alignment")
    aligned_tensor, l_kpts_moved = refine_tile(
            om_data,
            idx_tile=0,
            l_tiles=[rect_tile],
            l_cell_masks=l_cell_masks,
            overwrite_cache=True,
            out_prefix=out_prefix
    )
    dict_disp = torch.load(f"{dir_nonrigid}/{group_id}/roi_{i_box}/epoch3-0-4.ckpt")

    dict_IHC_layer = om_data.proj_info.load_IHC_info()
    gene_idx = np.array(dict_IHC_layer["order_label"])

    print("converting single cell h5ad")
    l_tensors = []
    for i_layer in range(1, 5):
        da_IHC = get_image_ihc(om_data, i_layer=i_layer)
        # y_beg, x_beg, y_end, x_end = l_tiles[i_box]
        disp = dict_disp[f"{i_layer}.displacement_field"]
        tensor_nonrigid = warp_tensor(om.tl.im2tensor(da_IHC[y_beg:y_end, x_beg:x_end, gene_idx]), disp.cpu(), **kwargs)
        l_tensors.append(tensor_nonrigid[:, 1:7, :, :])

    tensor_tile =  torch.cat(l_tensors, dim=1)
    adata = extract_cell_feats_h5ad(gdf_sub, tensor_tile, x_beg, y_beg)
    adata.var["gene"] = l_names
    adata.var_names = l_names
    adata.write_h5ad(f"{dir_nonrigid}/{group_id}/roi_{i_box}/TME.h5ad")
    fig = sc.pl.spatial(adata, color=l_names, spot_size=10, ncols=6, return_fig=True, show=False)
    # fig.savefig(f"{dir_nonrigid}/{group_id}/roi_{i_box}/TME.png")

    return adata


def main():
    group_id = sys.argv[1]
    
    os.chdir("/cluster/home/bqhu_jh/projects/panlab/code/wujiangchao/ALLTLS")
    with open("./config/panlab/config_pdac.yaml", 'r') as f:
        template_string = f.read()
        config_info = yaml.load(template_string, Loader=yaml.FullLoader)

    config_info["datasets"]["group"] = f"{group_id}"

    om_data = om.Omni3D(config_info=config_info)
    om_data.raw_is_tiff = True
    om_data.set_zoom_level(1)

    df_bboxes = pd.read_csv(f"./fig/cell_density/pos_{group_id}.csv")
    df_bboxes["area"] = (df_bboxes["x_end"] - df_bboxes["x_beg"]) * (df_bboxes["y_end"] - df_bboxes["y_beg"])
    df_bboxes["density"] = df_bboxes["score"] / df_bboxes["area"]

    N_boxes = min(df_bboxes.shape[0], 5)
    for i_box in tqdm(range(N_boxes), desc=f"Processing bounding boxes for {group_id}"):
        try:
            process_bbox(
                            om_data,
                            i_box,
                            df_bboxes,
            )
        except Exception as e:
            print(f"Error processing bounding box {i_box}: {e}")
            continue

if __name__ == "__main__":
    main()