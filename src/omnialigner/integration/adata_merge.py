import os
from typing import List, Tuple

import yaml
import pandas as pd
import scanpy as sc
import torch
import numpy as np
import cv2
from tqdm import tqdm
import dask.array as da
from sklearn.cluster import KMeans

import omnialigner as om
from omnialigner.dtypes import Tensor_kpts_N_xy, Np_image_HWC, Dask_image_HWC
from omnialigner.logging import logger as logging

class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        _, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected



def modality_integration(om_data: om.Omni3D):
    from omnialigner.integration.MISO import integrate_MISO
    z_HE_index, positions = find_p123_positions(om_data.proj_info.df_proj_info["type"])
    adata, adata_merged = make_adata_pairs(om_data, anchor_pos=z_HE_index, l_omics_pos=positions)
    outdir = f"{om_data.proj_info.root_dir}/analysis/{om_data.proj_info.project}/{om_data.proj_info.version}/08.MISO_3d/{om_data.proj_info.group}"
    clusters, _ = integrate_MISO(adata_merged.X, adata.X, outdir)

    adata_merged.obs["kmeans"] = [ str(x) for x in clusters]
    adata.write_h5ad(f"{outdir}/HE.h5ad")
    adata_merged.write_h5ad(f"{outdir}/IHC.h5ad")
    return adata, adata_merged


def cross_sample_HE_integration(
        config_yaml:str,
        l_group_ids: List[str],
        kmeans_n:int=10,
        overwrite_cache=False,
        outdir:str="./"
    ) -> sc.AnnData:
    if not overwrite_cache and os.path.exists(f"{outdir}/merged_adata_conch.h5ad"):
        adata = sc.read_h5ad(f"{outdir}/merged_adata_conch.h5ad")
        return adata

    l_adata = []
    for group_id in tqdm(l_group_ids, desc="Loading data"):
        with open(config_yaml, 'r', encoding="utf-8") as f:
            template_string = f.read()
            config_info = yaml.load(template_string, Loader=yaml.FullLoader)

        config_info["datasets"]["group"] = f"{group_id}"
        om_data = om.Omni3D(config_info=config_info)
        adata, _ = om.external.run_trident(om_data, i_layer=0, overwrite_cache=False, crop=True, target_crop_zoomlevel=1)
        adata.obs["label"] = group_id
        l_adata.append(adata)

    adata = sc.concat(l_adata)
    logging.info(f"Concatenated Done. PCA for {len(l_adata)} samples with {adata.shape[0]} regions and {adata.shape[1]} features.")
    x = torch.from_numpy(adata.X).float()
    fit_pca = TorchPCA(n_components=3).fit(x)
    x_red = fit_pca.transform(x)
    adata.obsm["pca_raw_rgb"] = ( (torch.clip(x_red, -5, 5) + 5) * 25.5).int().numpy()

    logging.info("harmony_integrate Start.")
    sc.tl.pca(adata, svd_solver="arpack")
    sc.external.pp.harmony_integrate(adata, key="label")
    x = torch.from_numpy(adata.obsm["X_pca_harmony"]).float()
    logging.info(f"harmony_integrate Done. PCA for {len(l_adata)} samples with {adata.shape[0]} regions and {adata.shape[1]} features.")

    fit_pca = TorchPCA(n_components=3).fit(x)
    x_red = fit_pca.transform(x)
    adata.obsm["pca_harmony_rgb"] = ( (torch.clip(x_red, -5, 5) + 5) * 25.5).int().numpy()
    X = adata.obsm["pca_harmony_rgb"]
    kmeans = KMeans(n_clusters=kmeans_n, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X)
    adata.obs[f"kmeans_{kmeans_n}"] = pd.Categorical(labels.astype(str))
    adata.write_h5ad(f"{outdir}/merged_adata_conch.h5ad")
    return adata

def make_HE_IHC_pairs(
        om_yaml:str,
        group_id:str,
        adata_sub: sc.AnnData,
        anchor_pos: int,
        l_omics_pos: List[int]
    ) -> Tuple[sc.AnnData, sc.AnnData]:
    with open(om_yaml, 'r', encoding="utf-8") as f:
        template_string = f.read()
        config_info = yaml.load(template_string, Loader=yaml.FullLoader)
    
    if group_id is not None:
        config_info["datasets"]["group"] = f"{group_id}"
    
    om_data = om.Omni3D(config_info=config_info)
    if len(om_data) < 5:
        return None, None


    offset_kpts = np.array([5, 8])
    
    scale_level = om_data.l_scales[3]
    ratio = om_data.l_scales[1] / om_data.l_scales[3]
    kpt_raw = torch.from_numpy(adata_sub.obsm["spatial"]).float().contiguous() / ratio
    om_data.set_zoom_level(3)
    kpt_nonrigid = om.align.apply_nonrigid_landmarks_HD(om_data, i_layer=anchor_pos, zoom_level=3, keypoints_raw=kpt_raw)
    kpt_nonrigid = (kpt_nonrigid.numpy() + offset_kpts) * scale_level 

    om_data.set_zoom_level(1)
    l_dask_rmaf = []
    for pos in l_omics_pos:
        da_1 = om.align.apply_nonrigid_HD(om_data, i_layer=pos, overwrite_cache=False, tag="rmaf").compute()
        l_dask_rmaf.append(da_1)
    
    dict_IHC_layer = om_data.proj_info.load_IHC_info()
    gene_idx = np.array(dict_IHC_layer["order_label"])[1:7]
    da_exp = da.concatenate([
        cv2.resize(x, (om_data.max_size, om_data.max_size))[:, :, gene_idx] for x in l_dask_rmaf
    ], axis=2)
    kpt_nonrigid_ = kpt_nonrigid / scale_level
    np_exp_kpts = extract_patches(da_exp.compute(), kpt_nonrigid_, patch_size=25)

    l_names = list(np.array(dict_IHC_layer["genes"]["P1"]))[1:7] +\
                list(np.array(dict_IHC_layer["genes"]["P2"]))[1:7] +\
                list(np.array(dict_IHC_layer["genes"]["P3"]))[1:7] +\
                list(np.array(dict_IHC_layer["genes"]["P4"]))[1:7]

    adata_merged = sc.AnnData(X=np_exp_kpts, obs=pd.DataFrame({"x": kpt_nonrigid_[:, 0], "y": kpt_nonrigid_[:, 1]}), var=pd.DataFrame({"gene_name": l_names}))
    adata_merged.obs["kmeans_10_HE"] = list(adata_sub.obs["kmeans_10"])
    adata_merged.obsm["spatial"] = kpt_nonrigid_
    adata_merged.var_names = l_names
    adata_merged.obs['label'] = group_id
    return adata_sub, adata_merged

def cross_sample_integration(
        om_yaml:str,
        l_group_ids: List[str],
        anchor_pos: int,
        l_omics_pos: List[int],
        outdir:str = "./",
        overwrite_cache:bool=False
    ) -> Tuple[sc.AnnData, sc.AnnData]:
    if os.path.exists(f"{outdir}/merged_adata_conch_ihc.h5ad") and not overwrite_cache:
        adata = sc.read_h5ad(f"{outdir}/merged_adata_conch_ihc.h5ad")
        return adata

    from omnialigner.integration.MISO import integrate_MISO
    l_adata_he: List[sc.AnnData] = []
    l_adata_ihc: List[sc.AnnData] = []
    adata = cross_sample_HE_integration(om_yaml, l_group_ids, overwrite_cache=overwrite_cache)    
    for sample_name in l_group_ids:
        adata_sub = adata[adata.obs["label"] == sample_name]
        adata_he, adata_ihc = make_HE_IHC_pairs(om_yaml, sample_name, adata_sub, anchor_pos, l_omics_pos)
        if adata_ihc is None:
            continue

        l_adata_ihc.append(adata_ihc)
        l_adata_he.append(adata_he)


    for i, ad in enumerate(l_adata_ihc):
        if not ad.var_names.is_unique:
            ad.var_names_make_unique()

    adata_he = sc.concat(l_adata_he)
    adata_ihc = sc.concat(l_adata_ihc)

    clusters, _ = integrate_MISO(np.log(adata_ihc.X+1),  adata_he.obsm["pca_harmony_rgb"], outdir=outdir, n_clusters=15)
    adata_ihc.obs["kmeans_15_miso"] = clusters
    adata_ihc.obsm["pca_harmony_rgb"] = adata_he.obsm["pca_harmony_rgb"]
    adata_ihc.write_h5ad(f"{outdir}/merged_adata_conch_ihc.h5ad")
    return adata_ihc


def extract_patches(image: Np_image_HWC|Dask_image_HWC, kpts:Tensor_kpts_N_xy, patch_size=10):
    """
    从图像中提取每个关键点附近的像素值。

    参数:
    - image: 输入图像，形状为 (H, W, C)
    - kpts: 关键点列表，每个关键点是 [x, y]
    - patch_size: 半径大小（默认为10）

    返回:
    - patches: 包含每个关键点周围像素值的列表，形状为 (2*patch_size+1, 2*patch_size+1, C)
    """
    H, W, _ = image.shape
    patches = []
    for idx, (x, y) in tqdm(enumerate(kpts)):
        x = int(round(x))
        y = int(round(y))
        x_min = max(x - patch_size, 0)
        x_max = min(x + patch_size + 1, W)
        y_min = max(y - patch_size, 0)
        y_max = min(y + patch_size + 1, H)
        patch = image[y_min:y_max, x_min:x_max, :]
        patches.append(patch.mean(axis=(0,1)))

    return np.stack(patches)


def make_adata_pairs(om_data: om.Omni3D, anchor_pos: int, l_omics_pos: List[int]) -> Tuple[sc.AnnData, sc.AnnData]:
    output_dir = f"{om_data.proj_info.root_dir}/analysis/{om_data.proj_info.project}/{om_data.proj_info.version}/02.1.dino_feats/{om_data.proj_info.group}/h5ad/"
    adata = sc.read_h5ad(f"{output_dir}/{anchor_pos}.h5ad")
    offset_kpts = np.array([5, 8])

    scale_level = om_data.l_scales[3]
    kpt_raw = torch.from_numpy(adata.obsm["spatial"]).float().contiguous()
    om_data.set_zoom_level(3)
    kpt_nonrigid = om.align.apply_nonrigid_landmarks_HD(om_data, i_layer=anchor_pos, zoom_level=3, keypoints_raw=kpt_raw)
    kpt_nonrigid = (kpt_nonrigid.numpy() + offset_kpts) * scale_level

    om_data.set_zoom_level(1)
    l_dask_rmaf = []
    for pos in l_omics_pos:
        da_1 = om.align.apply_nonrigid_HD(om_data, i_layer=pos, overwrite_cache=False, tag="rmaf").compute()
        l_dask_rmaf.append(da_1)
    
    dict_IHC_layer = om_data.proj_info.load_IHC_info()
    gene_idx = np.array(dict_IHC_layer["order_label"])[1:7]
    da_exp = da.concatenate([
        cv2.resize(x, (om_data.max_size, om_data.max_size))[:, :, gene_idx] for x in l_dask_rmaf
    ], axis=2)
    kpt_nonrigid_ = kpt_nonrigid / scale_level
    np_exp_kpts = extract_patches(da_exp.compute(), kpt_nonrigid_, patch_size=2)

    l_names = list(np.array(dict_IHC_layer["genes"]["P1"]))[1:7] +\
                list(np.array(dict_IHC_layer["genes"]["P2"]))[1:7] +\
                list(np.array(dict_IHC_layer["genes"]["P3"]))[1:7] +\
                list(np.array(dict_IHC_layer["genes"]["P4"]))[1:7]

    adata_merged = sc.AnnData(X=np_exp_kpts, obs=pd.DataFrame({"x": kpt_nonrigid_[:, 0], "y": kpt_nonrigid_[:, 1]}), var=pd.DataFrame({"gene_name": l_names}))
    adata_merged.obs["leiden_HE"] = adata.obs["leiden"]
    adata_merged.obsm["spatial"] = kpt_nonrigid_
    adata_merged.var_names = l_names
    return adata, adata_merged


def find_p123_positions(lst, anchor="HE"):
    result = []
    anchor_pos = []
    for i, val in enumerate(lst):
        if val == anchor:
            anchor_pos.append(i)
            pos_p1 = None
            pos_p2 = None
            pos_p3 = None
            dist = 1
            while None in [pos_p1, pos_p2, pos_p3]:
                right = i + dist
                if right < len(lst):
                    if lst[right] == "P1" and pos_p1 is None:
                        pos_p1 = right
                    elif lst[right] == "P2" and pos_p2 is None:
                        pos_p2 = right
                    elif lst[right] == "P3" and pos_p3 is None:
                        pos_p3 = right

                left = i - dist
                if left >= 0:
                    if lst[left] == "P1" and pos_p1 is None:
                        pos_p1 = left
                    elif lst[left] == "P2" and pos_p2 is None:
                        pos_p2 = left
                    elif lst[left] == "P3" and pos_p3 is None:
                        pos_p3 = left
                
                dist += 1

            result.append([pos_p1, pos_p2, pos_p3])

    return result, anchor_pos