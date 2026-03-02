"""
Benchmark script for different 3D alignment methods
"""
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import anndata as ad
import torch
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
import yaml
import cv2
import torch.nn.functional as F

import omnialigner as om
from alignment_utils import (
    get_sample_config,
    random_sample,
    load_or_create_adata,
    prepare_slices,
    kpt_to_png,
    scale_coords,
    assign_kpt_interpolate,
    adata_feature_to_rgb
)
from align3d_omni import run_omnialigner_alignment
# from align3d_omnialigner import run_omnialigner_alignment

sns = om.pl.sns


def func_sort_h5ads(algo):
    if algo == "spateo" or "STalign":
        func_ = lambda x: int(x.split("/")[-1].split("_")[-1].split(".h5ad")[0])
    
    if algo == "cpd" or algo == "loki":
        func_ = lambda x: int(x.split("/")[-1].split("_")[-2])

    return func_

def run_cpd_alignment(sample=None, device='cuda'):
    """Run CPD (Coherent Point Drift) alignment"""
    from pycpd import DeformableRegistration
    
    # sample, h5ad_path, out_root, label_col, _ = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    ref_layer = config_dict.get("ref_layer", 0)
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running CPD alignment for {sample}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    
    # Load data
    adata = load_or_create_adata(**config_dict)
    slides_raw, coords_t, z_height = prepare_slices(adata)
    
    # Prepare subsampled slices
    file_subidx = os.path.join(out_root, 'cpd', f'sub_node_idxs.pth')
    os.makedirs(os.path.join(out_root, 'cpd'), exist_ok=True)
    
    if not os.path.exists(file_subidx):
        dict_idx = {z: random_sample(coords, 20_000, seed_t=42) for z, coords in coords_t.items()}
        torch.save(dict_idx, file_subidx)
    
    dict_idx = torch.load(file_subidx)
    aligned_slices = [adata[adata.obsm['spatial_3D'][:, 2] == z][dict_idx[z]] for z in z_height]
    
    # Initialize first slice
    aligned_slices[0].obsm['spatial_2D'] = aligned_slices[0].obsm['spatial_3D'][:, :2]
    aligned_slices[0].obsm['align_spatial'] = aligned_slices[0].obsm['spatial_2D'].copy()
    
    if not os.path.exists(os.path.join(out_root, 'cpd', f'aligned_slice_0_lowrank.h5ad')):
        aligned_slices[0].write_h5ad(os.path.join(out_root, 'cpd', f'aligned_slice_0_lowrank.h5ad'))
    
    # Align remaining slices
    for i in tqdm(range(1, len(aligned_slices))):
        if os.path.exists(os.path.join(out_root, 'cpd', f'aligned_slice_{i}_lowrank.h5ad')):
            adata_ = ad.read_h5ad(os.path.join(out_root, 'cpd', f'aligned_slice_{i}_lowrank.h5ad'))
            aligned_slices[i].obsm['align_spatial'] = adata_.obsm['align_spatial']
            continue
        
        aligned_slices[i].obsm['spatial_2D'] = aligned_slices[i].obsm['spatial_3D'][:, :2].copy()
        
        np_X = np.array(aligned_slices[i-1].obsm['align_spatial'])
        np_Y = np.array(aligned_slices[i].obsm['spatial_2D'])
        
        reg = DeformableRegistration(X=np_X, Y=np_Y, max_iterations=5, low_rank=True)
        TY, (G, W) = reg.register()
        
        aligned_slices[i].obsm['align_spatial'] = TY
        aligned_slices[i].write_h5ad(os.path.join(out_root, 'cpd', f'aligned_slice_{i}_lowrank.h5ad'))
        torch.save({"G": G, "W": W}, os.path.join(out_root, 'cpd', f'params_{i}_lowrank.pth'))
    
    print(f"CPD alignment completed for {sample}")


def run_loki_alignment(sample=None, device='cuda'):
    """Run LOKI alignment (CPD with different parameters)"""
    from pycpd import DeformableRegistration
    from loki.align import align_tissue
    from loki.utils import get_pca_by_fit
    # sample, h5ad_path, out_root, label_col, _ = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    ref_layer = config_dict.get("ref_layer", 0)
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running LOKI alignment for {sample}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    
    # Load data
    adata = load_or_create_adata(**config_dict)
    slides_raw, coords_t, z_height = prepare_slices(adata)
    
    # Prepare subsampled slices
    file_subidx = os.path.join(out_root, 'loki', f'sub_node_idxs.pth')
    os.makedirs(os.path.join(out_root, 'loki'), exist_ok=True)
    
    if not os.path.exists(file_subidx):
        dict_idx = {z: random_sample(coords, 20_000, seed_t=42) for z, coords in coords_t.items()}
        torch.save(dict_idx, file_subidx)
    
    dict_idx = torch.load(file_subidx)
    aligned_slices = [adata[adata.obsm['spatial_3D'][:, 2] == z][dict_idx[z]] for z in z_height]
    
    # Initialize first slice
    aligned_slices[0].obsm['spatial_2D'] = aligned_slices[0].obsm['spatial_3D'][:, :2]
    aligned_slices[0].obsm['align_spatial'] = aligned_slices[0].obsm['spatial_2D'].copy()
    
    # if not os.path.exists(os.path.join(out_root, 'loki', f'aligned_slice_0_lowrank.h5ad')):
    #     aligned_slices[0].write_h5ad(os.path.join(out_root, 'loki', f'aligned_slice_0_lowrank.h5ad'))
    aligned_slices[0].write_h5ad(os.path.join(out_root, 'loki', f'aligned_slice_0_lowrank.h5ad'))
    
    # Align remaining slices
    for i in tqdm(range(1, len(aligned_slices))):
        # if os.path.exists(os.path.join(out_root, 'loki', f'aligned_slice_{i}_lowrank.h5ad')):
        #     adata_ = ad.read_h5ad(os.path.join(out_root, 'loki', f'aligned_slice_{i}_lowrank.h5ad'))
        #     aligned_slices[i].obsm['align_spatial'] = adata_.obsm['align_spatial']
        #     continue
        
        if "txt_features" in aligned_slices[i-1].obsm:
            tar_features = aligned_slices[i-1].obsm["txt_features"]
            src_features = aligned_slices[i].obsm["txt_features"]
        else:
            tar_features = aligned_slices[i-1].X if isinstance(aligned_slices[i-1].X, np.ndarray) else aligned_slices[i-1].X.toarray()
            src_features = aligned_slices[i].X if isinstance(aligned_slices[i].X, np.ndarray) else aligned_slices[i].X.toarray()

        pca_comb_features, _ = get_pca_by_fit(tar_features.T, src_features.T)
        ad_tar_coor = np.array(aligned_slices[i-1].obsm['align_spatial'])
        ad_src_coor = np.array(aligned_slices[i].obsm['spatial_2D'])
        src_img = np.zeros((h, w, 3), dtype=np.uint8)
        cpd_coor, homo_coor, _ = align_tissue(ad_tar_coor, ad_src_coor, pca_comb_features, src_img)
        
        aligned_slices[i].obsm['align_spatial'] = homo_coor
        aligned_slices[i].write_h5ad(os.path.join(out_root, 'loki', f'aligned_slice_{i}_lowrank.h5ad'))

    
    print(f"LOKI alignment completed for {sample}")


def run_moscot_alignment(sample=None, device='cuda'):
    """Run MOSCOT alignment"""
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    
    import moscot as mt
    from moscot.problems.space import AlignmentProblem
    
    # sample, h5ad_path, out_root, label_col, _ = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    ref_layer = config_dict.get("ref_layer", 0)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running MOSCOT alignment for {sample}")
    # Load data
    adata_raw = load_or_create_adata(**config_dict)
    slides_raw, coords_t, z_height = prepare_slices(adata_raw)
    
    # Run MOSCOT alignment
    col_layer = "i_layer"
    l_layers = sorted(adata_raw.obs[col_layer].unique())
    # output_path = os.path.join(out_root, 'moscot', f'aligned_slice_{l_layers[-1]}.h5ad')
    # if os.path.exists(output_path):
    #     print(f"Loading existing MOSCOT alignment from {output_path}")
    #     return 
    

    file_subidx = os.path.join(out_root, 'moscot', f'sub_node_idxs.pth')
    os.makedirs(os.path.join(out_root, 'moscot'), exist_ok=True)
    if not os.path.exists(file_subidx):
        n_subsample = 5_000
        if sample=="abca-1":
            n_subsample = 2_000

        dict_idx = {z: random_sample(coords, n_subsample, seed_t=42) for z, coords in coords_t.items()}
        torch.save(dict_idx, file_subidx)
    
    dict_idx = torch.load(file_subidx)
    aligned_slices: List[sc.AnnData] = [adata_raw[adata_raw.obsm['spatial_3D'][:, 2] == z][dict_idx[z]] for z in z_height]
    adata = sc.concat(aligned_slices)
    adata.obs[col_layer] = pd.Categorical(adata.obs[col_layer], categories=l_layers, ordered=True)
    ap = AlignmentProblem(adata=adata)
    ap = ap.prepare(batch_key=col_layer, policy="sequential", spatial_key="spatial_2D")
    ap = ap.solve(batch_size=512, device=device)
    
    df_col = adata.obs[col_layer]
    first_layer = df_col.cat.categories[0] if df_col.dtype == "category" else sorted(df_col.unique())[0]
    ap.align(reference=first_layer, key_added="align_spatial_nonrigid")
    
    # Save results
    os.makedirs(os.path.join(out_root, 'moscot'), exist_ok=True)
    for idx, i_layer in enumerate(l_layers):
        adata_slice = adata[adata.obs[col_layer] == i_layer]
        adata_slice.obsm["align_spatial"] = adata_slice.obsm["align_spatial_nonrigid"]
        adata_slice.write_h5ad(os.path.join(out_root, 'moscot', f'aligned_slice_{idx}.h5ad'))
        
    print(f"MOSCOT alignment completed for {sample}")


def run_spateo_alignment(sample=None, device='cuda'):
    """Run Spateo alignment"""
    import spateo as st
    
    # sample, h5ad_path, out_root, label_col, _ = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    ref_layer = config_dict.get("ref_layer", 0)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running Spateo alignment for {sample}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    print(f"Spateo version: {st.__version__}")
    
    # Load data
    
    adata = load_or_create_adata(**config_dict)
    slides_raw, coords_t, z_height = prepare_slices(adata)
    
    # Prepare subsampled slices
    spatial_key = 'spatial_2D'
    n_subsample = 20_000
    if sample=="abca-1" or sample=="mouse_E11.5_embryo":
        n_subsample = 5_000
    dict_idx = {z: random_sample(coords, n_subsample, seed_t=42) for z, coords in coords_t.items()}
    
    os.makedirs(os.path.join(out_root, 'spateo'), exist_ok=True)
    torch.save(dict_idx, os.path.join(out_root, 'spateo', f'sub_node_idxs.pth'))
    
    slices = [adata[adata.obsm['spatial_3D'][:, 2] == z][dict_idx[z]] for z in z_height]
    slices[0].obsm['spatial_2D'] = slices[0].obsm['spatial_3D'][:, :2]
    









    
    for i in range(1, len(slices)):
        slices[i].obsm['spatial_2D'] = slices[i].obsm['spatial_3D'][:, :2].copy()
        st.align.rigid_transformation(slices[i], spatial_key='spatial_2D', key_added='spatial_2D', theta=0.0)
    
    # Run alignment
    key_added = 'align_spatial'

















    aligned_slices, pis = st.align.morpho_align(
        models=slices,
        spatial_key=spatial_key,
        key_added=key_added,
        device=device,
        # nonrigid related
        beta=1,
        lambdaVF=1,
        K=30,
        # sparse and chunk calculation
        sparse_calculation_mode=True,
        use_chunk=True,
        chunk_capacity=4,
        verbose=True,
        # use PCA feature
        rep_layer='X_pca',
        rep_field='obsm',
        dissimilarity='cos',
    )
    
    # Save results
    for i, aligned_slice in enumerate(aligned_slices):
        # aligned_slice.uns["transformation"] = transformation[i]
        if "iter_spatial" in aligned_slice.uns:
            for key in ["align_spatial", "sigma2"]:
                aligned_slice.uns["iter_spatial"][key] = {str(k): v for k, v in aligned_slice.uns["iter_spatial"][key].items()}
        aligned_slice.write_h5ad(os.path.join(out_root, 'spateo', f'aligned_slice_{i}.h5ad'))
    
    print(f"Spateo alignment completed for {sample}")


def run_stalign_alignment(sample=None, device='cuda'):
    """Run STalign alignment"""
    from STalign import STalign
    
    # sample, h5ad_path, out_root, _ = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    ref_layer = config_dict.get("ref_layer", 0)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running STalign alignment for {sample}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    
    def stalign_lddmm_alignment(coord_I, coord_J, I, J, params=None):
        """STalign LDDMM alignment helper"""
        if not isinstance(coord_I, (tuple, list)) or len(coord_I) != 2:
            raise ValueError("coord_I must be a tuple/list containing xI,yI")
        if not isinstance(coord_J, (tuple, list)) or len(coord_J) != 2:
            raise ValueError("coord_J must be a tuple/list containing xJ,yJ")
        if not isinstance(I, np.ndarray) or not isinstance(J, np.ndarray):
            raise ValueError("I and J must be numpy arrays")
        
        xI, yI = coord_I
        xJ, yJ = coord_J
        
        default_params = {}
        if params is not None:
            default_params.update(params)
        params = default_params
        
        YI, XI = np.array(yI), np.array(xI)
        YJ, XJ = np.array(yJ), np.array(xJ)
        
        out = STalign.LDDMM([YI, XI], I, [YJ, XJ], J, **params)
        
        xv, v, A = out['xv'], out['v'], out['A']
        xv = [xx.detach().cpu() for xx in xv]
        v = v.detach().cpu()
        A = A.detach().cpu()
        
        phii = STalign.build_transform(xv, v, A, XJ=[YJ, XJ], direction='b')
        phiI = STalign.transform_image_source_to_target(xv, v, A, [YI, XI], I, [YJ, XJ])
        
        out["xv"] = xv
        out["v"] = v
        out["A"] = A
        
        return (phii, phiI), out
    
    def align_two_slices_stalign(adata_ref, adata_query, device='cuda'):
        """Align two slices using STalign"""
        params = {
            'niter': 1_000,
            'device': device,
            'epV': 50
        }
        xI, yI = adata_query.obsm['spatial_2D'][:, 0], adata_query.obsm['spatial_2D'][:, 1]
        xJ, yJ = adata_ref.obsm['align_spatial'][:, 0], adata_ref.obsm['align_spatial'][:, 1]
        
        XI, YI, I = STalign.rasterize(xI, yI, dx=30, blur=1.5, draw=False)
        XJ, YJ, J = STalign.rasterize(xJ, yJ, dx=30, blur=1.5, draw=False)
        (phii, phiI), out = stalign_lddmm_alignment((XI, YI), (XJ, YJ), I, J, params)
        
        tpointsI = STalign.transform_points_source_to_target(
            out['xv'], out['v'], out['A'], 
            np.stack([yI, xI], 1)
        ).cpu().numpy()[:, ::-1]
        return tpointsI
    
    # Load data
    adata = load_or_create_adata(**config_dict)
    slides_raw, coords_t, z_height = prepare_slices(adata)
    
    # Prepare slices
    os.makedirs(os.path.join(out_root, 'STalign'), exist_ok=True)
    aligned_slices = [adata[adata.obsm['spatial_3D'][:, 2] == z] for z in z_height]
    
    # Initialize first slice
    aligned_slices[0].obsm['spatial_2D'] = aligned_slices[0].obsm['spatial_3D'][:, :2]
    aligned_slices[0].obsm['align_spatial'] = aligned_slices[0].obsm['spatial_2D'].copy()
    aligned_slices[0].write_h5ad(os.path.join(out_root, 'STalign', f'aligned_slice_0.h5ad'))
    # if not os.path.exists(os.path.join(out_root, 'STalign', f'aligned_slice_0.h5ad')):
    #     aligned_slices[0].write_h5ad(os.path.join(out_root, 'STalign', f'aligned_slice_0.h5ad'))
    
    # Align remaining slices
    for i in tqdm(range(1, len(aligned_slices))):
        # if os.path.exists(os.path.join(out_root, 'STalign', f'aligned_slice_{i}.h5ad')):
        #     adata_ = ad.read_h5ad(os.path.join(out_root, 'STalign', f'aligned_slice_{i}.h5ad'))
        #     aligned_slices[i].obsm['align_spatial'] = adata_.obsm['align_spatial']
        #     continue
        
        aligned_slices[i].obsm['spatial_2D'] = aligned_slices[i].obsm['spatial_3D'][:, :2].copy()
        TY = align_two_slices_stalign(aligned_slices[i-1], aligned_slices[i], device=device)
        aligned_slices[i].obsm['align_spatial'] = TY
        aligned_slices[i].write_h5ad(os.path.join(out_root, 'STalign', f'aligned_slice_{i}.h5ad'))
    
    print(f"STalign alignment completed for {sample}")

def run_SPACEL_alignment(sample=None, device='cuda'):
    """Run SPACEL alignment"""
    from SPACEL import Scube

    # sample, h5ad_path, out_root, label_col, _ = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    ref_layer = config_dict.get("ref_layer", 0)
    os.makedirs(out_root, exist_ok=True)
    
    print(f"Running SPACEL alignment for {sample}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    adata = load_or_create_adata(**config_dict)
    slides_raw, coords_t, z_height = prepare_slices(adata)
    # if os.path.exists(os.path.join(out_root, 'SPACEL', f'aligned_slice_{len(z_height)-1}.h5ad')):
    #     print(f"Loading existing SPACEL alignment for {sample}")
    #     return 
    
    # Prepare subsampled slices
    file_subidx = os.path.join(out_root, 'SPACEL', f'sub_node_idxs.pth')
    os.makedirs(os.path.join(out_root, 'SPACEL'), exist_ok=True)
    
    if not os.path.exists(file_subidx):
        dict_idx = {z: random_sample(coords, 20_000, seed_t=42) for z, coords in coords_t.items()}
        torch.save(dict_idx, file_subidx)
    
    dict_idx = torch.load(file_subidx)
    # aligned_slices: List[sc.AnnData] = [adata[adata.obsm['spatial_3D'][:, 2] == z][dict_idx[z]] for z in z_height]
    aligned_slices: List[sc.AnnData] = []
    for z in z_height:
        adata_slice = adata[adata.obsm['spatial_3D'][:, 2] == z]
        if z in dict_idx:
            adata_slice = adata_slice[dict_idx[z]]
        
        adata_slice.obsm["spatial_2D"] = adata_slice.obsm['spatial_3D'][:, :2]
        aligned_slices.append(adata_slice)

    max_slices = max([adata_slice.n_obs for adata_slice in aligned_slices])
    subset_prop = 20_000 / max_slices if max_slices > 20_000 else None
    print(f"Using subset proportion: {subset_prop}")
    Scube.align(aligned_slices,
        raw_loc_key="spatial_2D",
        cluster_key=label_col,
        n_neighbors=4,
        p=1,
        n_threads=32,
        subset_prop=subset_prop,
        write_loc_path=os.path.join(out_root, 'SPACEL', 'aligned_coordinates.csv')
    )
    adata_concat_ = sc.concat(aligned_slices)
    extend_ratio = config_dict.get("extend_ratio", 0.2)
    adata_concat_.obsm["spatial_aligned_x"] = scale_coords(adata_concat_.obsm["spatial_aligned"].copy(), extent=extend_ratio) * np.array([w, h])
    l_layers = sorted(adata.obs["i_layer"].unique().tolist())
    for idx, i_layer in enumerate(l_layers):
        adata_slice_ = adata_concat_[adata_concat_.obs["i_layer"]==i_layer]
        aligned_slices[idx].obsm["align_spatial"] = adata_slice_.obsm["spatial_aligned_x"].values
        aligned_slices[idx].write_h5ad(os.path.join(out_root, 'SPACEL', f'aligned_slice_{idx}.h5ad'))


def concat_h5ad(sample, algo, using_color=False):
    """
        Concatenate aligned h5ad files and generate 3D image tensor
        
        Args:
            sample: sample name
            algo: alignment algorithm name
            using_color: whether to use color for visualization
        

        - obsm key: ** 'align_spatial'** is required for the input `glob(f"{out_root}/{algo}/align*.h5ad")` anndata objects

        - obsm key: **`align_spatial_nonrigid`** will be added to the output concatenated anndata object

        - when downsampling is applied during alignment, the aligned coordinates
        need to be interpolated back to the original points(adata.obsm['spatial']). This function handles that process.

        

    """
    def read_h5ad(file_path):
        adata = sc.read_h5ad(file_path)
        for tag in ['aligned_spatial', 'aligned_spatial_nonrigid', 'aligned_spatial_rigid']:
            if tag in adata.obsm:
                tag_renamed = tag.replace('aligned', 'align')
                adata.obsm[tag_renamed] = adata.obsm[tag].copy()
        
        return adata

    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    ref_layer = config_dict.get("ref_layer", 0)
    os.makedirs(out_root, exist_ok=True)
    out_file_img = f"{out_root}/{algo}/image_3d_tensor_final.pth" if using_color else f"{out_root}/{algo}/image_3d_category_nonrigid.pth"
    out_file_h5ad = f"{out_root}/{algo}/adata_concat.h5ad"

    adata = load_or_create_adata(**config_dict)
    # if os.path.exists(out_file_img) and os.path.exists(out_file_h5ad):
    #     print(f"3D image tensor already exists: {out_file_img} and {out_file_h5ad}")
    #     return
    
    dict_name = torch.load(f"{out_root}/color_dict.pth")

    
    l_h5ads = glob(f"{out_root}/{algo}/align*.h5ad")
    l_h5ads = sorted(l_h5ads, key=func_sort_h5ads(algo))
    l_adatas = []
    l_layers = sorted(adata.obs["i_layer"].unique().tolist())
    # for idx,i_layer in tqdm(enumerate(range(len(l_h5ads))), desc=f"Processing {algo} h5ads"):
    for idx,i_layer in tqdm(enumerate(l_layers), desc=f"Processing {algo} h5ads"):
        adata_sub = read_h5ad(l_h5ads[idx])
        adata_all = adata[adata.obs["i_layer"] == i_layer]
        adata_all = assign_kpt_interpolate(adata_sub, adata_all, key_common="spatial", key_map="align_spatial")
        l_adatas.append(adata_all)

    adata_concat = sc.concat(l_adatas)
    # adata_concat.obsm["align_spatial_nonrigid"] = scale_coords(adata_concat.obsm["align_spatial"], extent=0.1)
    adata_concat.obsm["align_spatial_nonrigid"] = adata_concat.obsm["align_spatial"]
    adata_concat.write_h5ad(out_file_h5ad)
    _h5ad_to_image_3d_tensor(sample, algo, using_color=using_color)    
    # l_imgs = []
    
    # for idx, i_layer in tqdm(enumerate(l_layers), total=len(l_layers), desc="Generating 3D image tensor"):
    #     adata_tmp = adata_concat[adata_concat.obs["i_layer"] == i_layer]
        
    #     l_clusters = [ int(l_name_used.index(v))+1 if v in l_name_used else -1 for v in adata_tmp.obs[label_col] ]
        
    #     l_colors = [ color_cells.get(i, "#888888") for i in l_clusters ]

    #     np_uint8 = kpt_to_png(adata_tmp.obsm["align_spatial_nonrigid"][:,0:2]*np.array([w, h]), l_colors, l_clusters=l_clusters, h=h, w=w, using_color=using_color, radius=radius)
    #     tensor = om.tl.im2tensor(np_uint8)
    #     l_imgs.append(tensor)

    # image_3d_tensor = torch.concat(l_imgs)*255
    # torch.save(image_3d_tensor, out_file_img)


def _h5ad_to_image_3d_tensor(sample, algo, using_color=False):
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    os.makedirs(out_root, exist_ok=True)
    out_file_img = f"{out_root}/{algo}/image_3d_tensor_final.pth" if using_color else f"{out_root}/{algo}/image_3d_category_nonrigid.pth"
    concat_h5ad_file = f"{out_root}/{algo}/adata_concat.h5ad"
    dict_name = torch.load(f"{out_root}/color_dict.pth")
    l_name_used = dict_name["l_name_used"]
    color_cells = dict_name["color_cells"]
    adata_concat = sc.read(concat_h5ad_file)
    l_layers = sorted(adata_concat.obs["i_layer"].unique().tolist())
    l_imgs = []
    for idx, i_layer in tqdm(enumerate(l_layers), total=len(l_layers), desc="Generating 3D image tensor"):
        adata_tmp = adata_concat[adata_concat.obs["i_layer"] == i_layer]
        l_clusters = [ int(l_name_used.index(v))+1 if v in l_name_used else -1 for v in adata_tmp.obs[label_col] ]
        l_colors = [ color_cells.get(i, "#888888") for i in l_clusters ]
        np_uint8 = kpt_to_png(adata_tmp.obsm["align_spatial_nonrigid"][:,0:2], l_colors, l_clusters=l_clusters, h=h, w=w, using_color=using_color, radius=radius)
        tensor = om.tl.im2tensor(np_uint8)
        l_imgs.append(tensor)

    image_3d_tensor = torch.concat(l_imgs)*255
    torch.save(image_3d_tensor, out_file_img)

def evaluate_registration(sample: str, algo: str, overwrite: bool=False) -> pd.DataFrame:
    """
    Calculate registration evaluation metrics (NMI, ARI) between consecutive layers after alignment.

    Steps:
    1. Load the 3D image tensor generated after alignment.
        If not exists, generate it from the aligned h5ad files using `adata.obsm['align_spatial_nonrigid']`.
    2. For each pair of consecutive layers, compute evaluation metrics (NMI, ARI).
    3. Store the results in a DataFrame and save to CSV.

    Args:
        sample: Sample name
        algo: Alignment algorithm name
        overwrite: Whether to overwrite existing evaluation results
    Returns:
        df: DataFrame containing evaluation metrics
    """
    # sample, h5ad_path, out_root, label_col, radius = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    out_file_img =  f"{out_root}/{algo}/image_3d_category_nonrigid.pth"
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    metric_funcs = {
        "nmi": normalized_mutual_info_score,
        "ari": adjusted_rand_score
    }

    
    # if os.path.exists(f"{out_root}/{algo}/registration_evaluation.csv") and not overwrite:
    #     df = pd.read_csv(f"{out_root}/{algo}/registration_evaluation.csv", index_col=False)
    #     return df

    dict_result = {"type": [], "func": [], "value": [], "layer": []}

    # if not os.path.exists(out_file_img):
    _h5ad_to_image_3d_tensor(sample, algo, using_color=False)

    np_raw = torch.load(out_file_img).cpu().numpy().astype(np.int32)
    for i_layer in tqdm(range(np_raw.shape[0]-1)):
        for metric_name, metric_func in metric_funcs.items():
            data_array = np_raw
            if i_layer+1 >= data_array.shape[0]:
                continue
            
            np_x = data_array[i_layer].ravel()
            np_y = data_array[i_layer+1].ravel()
            # idx  = (np_x>0) & (np_y>0)
            # np_x = np_x[idx]
            # np_y = np_y[idx]
            value = metric_func(np_x, np_y)
            dict_result["type"].append(algo)
            dict_result["func"].append(metric_name)
            dict_result["value"].append(value)
            dict_result["layer"].append(i_layer)


    df = pd.DataFrame(dict_result)
    df.to_csv(f"{out_root}/{algo}/registration_evaluation.csv", index=False)
    return df


import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Dict, Optional

def mnn_matching(
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    features_1: Optional[np.ndarray] = None,
    features_2: Optional[np.ndarray] = None,
    use_features: bool = False
) -> np.ndarray:
    """
    Compute sample1->sample2 matching indices based on MNN (Mutual Nearest Neighbors)
    Prioritize feature matching (e.g. gene expression) over spatial coordinates
    
    Args:
        coords_1: Sample 1 spatial coordinates, [N1,2] (y/x)
        coords_2: Sample 2 spatial coordinates, [N2,2] (y/x)
        features_1: Sample 1 feature matrix (e.g. gene expression), [N1,C]
        features_2: Sample 2 feature matrix (e.g. gene expression), [N2,C]
        use_features: Whether to use features for MNN (otherwise use spatial coords)
    
    Returns:
        i_prime: Matching indices from sample 1 to sample 2, [N1,]
    """
    # Select matching basis (features/coordinates)
    data_1 = features_1 if (use_features and features_1 is not None) else coords_1
    data_2 = features_2 if (use_features and features_2 is not None) else coords_2
    
    # Build KD trees
    tree1 = KDTree(data_1)
    tree2 = KDTree(data_2)
    
    # Step 1: Sample1 -> Sample2 nearest neighbors (1-NN)
    dist1, idx1 = tree2.query(data_1, k=1)  # idx1: [N1,], nearest neighbor in sample2 for each point in sample1
    
    # Step 2: Sample2 -> Sample1 nearest neighbors (1-NN)
    dist2, idx2 = tree1.query(data_2, k=1)  # idx2: [N2,], nearest neighbor in sample1 for each point in sample2
    
    # Step 3: Filter mutual nearest neighbors (MNN)
    i_prime = np.full(len(coords_1), -1)  # Initialize matching indices (-1 means no MNN match)
    for i in range(len(coords_1)):
        j = idx1[i]  # Point i in sample1 has nearest neighbor j in sample2
        if idx2[j] == i:  # Point j in sample2 has nearest neighbor i in sample1 -> mutual nearest neighbor
            i_prime[i] = j
    
    # For points without MNN match, fallback to one-way nearest neighbor
    no_mnn_mask = (i_prime == -1)
    i_prime[no_mnn_mask] = idx1[no_mnn_mask]
    
    return i_prime

def calculate_clc_score_mnn(
    labels_1: np.ndarray,
    labels_2: np.ndarray,
    coords_1: np.ndarray,
    coords_2: np.ndarray,
    features_1: Optional[np.ndarray] = None,
    features_2: Optional[np.ndarray] = None,
    use_features_for_mnn: bool = True,
    k_percent: float = 2.5,
    return_details: bool = False
) -> float | Tuple[float, Dict]:
    """
    Calculate CLC (Contextual Label Consistency) score based on MNN matching
    CLC(M) = (1/N) * Σ[I(l1(i)=l2(i')) * (1/|N_i|) * Σ[d(i',j') for j in N_i]]
    
    Args:
        labels_1: Sample 1 label array, [N1,] (cell types/brain regions/etc.)
        labels_2: Sample 2 label array, [N2,] (semantically consistent with labels_1)
        coords_1: Sample 1 spatial coordinates, [N1,2] (y/x)
        coords_2: Sample 2 spatial coordinates, [N2,2] (y/x)
        features_1: Sample 1 feature matrix (e.g. gene expression), [N1,C] (for MNN matching)
        features_2: Sample 2 feature matrix (e.g. gene expression), [N2,C] (for MNN matching)
        use_features_for_mnn: Whether to use features for MNN computation (otherwise use spatial coords)
        k_percent: Neighborhood K value percentage (2.5% of average cell count)
        return_details: Whether to return intermediate results
    
    Returns:
        clc_score: CLC score (0~1)
        details: Detailed result dictionary (when return_details=True)
    """
    # ===================== Input validation =====================
    N1 = len(coords_1)
    N2 = len(coords_2)
    if len(labels_1) != N1 or len(labels_2) != N2:
        raise ValueError(f"Coordinate and label length mismatch: coords_1={N1}(labels_1={len(labels_1)}), coords_2={N2}(labels_2={len(labels_2)})")
    if use_features_for_mnn and (features_1 is None or features_2 is None):
        raise ValueError("When use_features_for_mnn=True, must provide features_1 and features_2")
    if k_percent <= 0 or k_percent > 100:
        raise ValueError(f"k_percent must be in (0,100] range, current value: {k_percent}")
    
    # ===================== 1. MNN matching to compute i' (core replacement of M matrix) =====================
    i_prime = mnn_matching(
        coords_1=coords_1,
        coords_2=coords_2,
        features_1=features_1,
        features_2=features_2,
        use_features=use_features_for_mnn
    )
    
    # ===================== 2. Calculate neighborhood K value =====================
    avg_cell_num = (N1 + N2) / 2
    k = max(1, int(avg_cell_num * k_percent / 100))  # At least 1 neighborhood point
    
    # ===================== 3. Build spatial neighborhoods (K-nearest neighbors) =====================
    # K-nearest neighbors for sample 1 (N_i)
    kdtree_1 = KDTree(coords_1)
    neighborhood_1 = []
    for i in range(N1):
        neighbors = kdtree_1.query(coords_1[i], k=k)[1]
        neighborhood_1.append(neighbors if isinstance(neighbors, np.ndarray) else [neighbors])
    
    # K-nearest neighbors for sample 2 (N'_i')
    kdtree_2 = KDTree(coords_2)
    neighborhood_2 = []
    for j in range(N2):
        neighbors = kdtree_2.query(coords_2[j], k=k)[1]
        neighborhood_2.append(neighbors if isinstance(neighbors, np.ndarray) else [neighbors])
    
    # ===================== 4. Calculate CLC score =====================
    individual_contributions = np.zeros(N1)
    for i in range(N1):
        # Step 1: Label consistency indicator function I(l1(i)=l2(i'))
        ip = i_prime[i]  # Point i in sample1 matches to ip in sample2
        if ip < 0 or ip >= N2:  # No valid match
            individual_contributions[i] = 0.0
            continue
        l1_i = labels_1[i]
        l2_ip = labels_2[ip]
        label_consistency = 1.0 if (l1_i == l2_ip) else 0.0
        if label_consistency == 0:
            individual_contributions[i] = 0.0
            continue
        
        # Step 2: Spatial neighborhood consistency (1/|N_i|) * Σd(i',j')
        Ni = neighborhood_1[i]
        d_vals = []
        for j in Ni:  # j is a neighborhood point of i in sample1
            jp = i_prime[j]  # j matches to jp in sample2 (j')
            if jp < 0 or jp >= N2:
                d_vals.append(0.0)
                continue
            # d(i',j') = 1 if jp ∈ N'_ip, otherwise 0
            d = 1.0 if (jp in neighborhood_2[ip]) else 0.0
            d_vals.append(d)
        spatial_consistency = np.mean(d_vals) if len(d_vals) > 0 else 0.0
        
        # Step 3: Individual point contribution
        individual_contributions[i] = label_consistency * spatial_consistency
    
    # Step 4: Global average to get CLC score
    clc_score = np.mean(individual_contributions)
    
    # ===================== Return results =====================
    if return_details:
        details = {
            "i_prime": i_prime,  # Sample1->Sample2 MNN matching indices
            "neighborhood_1": neighborhood_1,  # K-nearest neighbors for each point in sample1
            "neighborhood_2": neighborhood_2,  # K-nearest neighbors for each point in sample2
            "individual_contributions": individual_contributions,
            "k_value": k,
            "avg_cell_num": avg_cell_num,
            "mnn_matching_mask": (i_prime != -1)  # Mask for successfully matched MNN points
        }
        return clc_score, details
    return clc_score



def calculate_adata_clc_scores(sample, algo, basis: Dict[str, str]=None) -> pd.DataFrame:
    """
        Calculate CLC scores for aligned adata between adjacent layers:
        Args:
            sample: Sample name
            algo: Alignment algorithm name
            basis: Dictionary of `adata.obsm` specifying basis for CLC calculation, e.g. {"spatial":"raw", "align_spatial_nonrigid":"align"}
        Returns:
            DataFrame of CLC scores between adjacent layers
        
        Steps:
        1. For each pair of adjacent layers (i, i+1):
            a. Extract cluster labels and coordinates based on specified basis
            b. Compute CLC scores using MNN matching for each basis
        2. Aggregate CLC scores into a DataFrame and save to CSV
        3. Return the DataFrame of CLC scores
    """
    if basis is None:
        basis = {"spatial_3D":"raw", "align_spatial_nonrigid":"align"}

    # sample, h5ad_path, out_root, label_col, radius = get_sample_config(sample)
    config_dict = get_sample_config(sample)
    sample = config_dict.get("sample")
    h5ad_path = config_dict.get("h5ad_path")
    out_root = config_dict.get("out_root")
    label_col = config_dict.get("label_col")
    radius = config_dict.get("radius")
    h, w = config_dict.get("h", 1280), config_dict.get("w", 1280)
    adata = sc.read_h5ad(f"{out_root}/{algo}/adata_concat.h5ad")
    dct_color = torch.load(f"{out_root}/color_dict.pth")
    l_name_used = dct_color["l_name_used"]
    l_layers = sorted(adata.obs["i_layer"].unique().tolist())
    l_scores = []
    pbar = tqdm(enumerate(l_layers[:-1]), total=len(l_layers)-1)
    for i_layer, layer in pbar:
        adata_i = adata[adata.obs["i_layer"] == l_layers[i_layer]]
        adata_j = adata[adata.obs["i_layer"] == l_layers[i_layer+1]]
        np_clusters_i = np.array([ int(l_name_used.index(v))+1 if v in l_name_used else -1 for v in adata_i.obs[label_col] ])
        np_clusters_j = np.array([ int(l_name_used.index(v))+1 if v in l_name_used else -1 for v in adata_j.obs[label_col] ])
        l_lines = []
        for key in sorted(basis):
            if key not in adata_i.obsm.keys() or key not in adata_j.obsm.keys():
                raise ValueError(f"adata.obsm do not contains {key}")
            kpt_i = adata_i.obsm[key][:, 0:2]
            kpt_j = adata_j.obsm[key][:, 0:2]
            clc_score = calculate_clc_score_mnn(np_clusters_i, np_clusters_j, kpt_i, kpt_j, use_features_for_mnn=False)
            l_lines.append(clc_score)

        l_scores.append(l_lines)
        dict_line = { f"CLC_{key}": l_lines[idx] for idx, key in enumerate(sorted(basis)) }
        pbar.set_postfix(dict_line)

    
    df_clc = pd.DataFrame(l_scores, columns=[ f"CLC_{key}" for key in sorted(basis) ])
    df_clc.to_csv(f"{out_root}/{algo}/clc_scores.csv")
    return df_clc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run 3D alignment benchmarks.')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['cpd', 'loki', 'moscot', 'spateo', 'STalign', 'omnialigner', 'SPACEL', 'all'],
                        help='Alignment method to run')
    parser.add_argument('--sample', type=str, default='mouse_E9.5_embryo',
                        choices=['abca-1', 'abca-2', 'abca-3', 'mouse_E11.5_embryo', 'mouse_E9.5_embryo', 'stereo_seq_mouse_embryo', 'ST_mouse_brain', 'merfish_mouse_brain', "starmap_3d_mouse_brain", "SmallIntestine", "metastatic_lymph_node", "slideseq_brain"],
                        help='Sample to process')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    methods = {
        'loki': run_loki_alignment,
        'moscot': run_moscot_alignment,
        'spateo': run_spateo_alignment,
        'STalign': run_stalign_alignment,
        'SPACEL': run_SPACEL_alignment,
        'omnialigner': run_omnialigner_alignment,
    }

    dict_basis = {
        'spatial_3D' : "raw", 
        'align_spatial_stacked': "stacked",
        'align_spatial_affine': "affine",
        'align_spatial_nonrigid': "nonrigid",
        'spatial_ref': "ref",
    }
    device = args.device
    device = "cpu" if not torch.cuda.is_available() else device
    if args.method == 'all':
        for method_name, method_func in methods.items():
            print(f"\n{'='*50}")
            print(f"Running {method_name.upper()}")
            print(f"{'='*50}\n")
            method_func(args.sample, device)
            basis = dict_basis.copy()
            if method_name != 'omnialigner':
                basis = None
                concat_h5ad(args.sample, method_name, using_color=False)

            calculate_adata_clc_scores(args.sample, method_name, basis=basis)
            evaluate_registration(args.sample, method_name, overwrite=False)
        
    else:
        methods[args.method](args.sample, device)
        basis = dict_basis.copy()
        if args.method != 'omnialigner':
            basis = None
            concat_h5ad(args.sample, args.method, using_color=False)

        calculate_adata_clc_scores(args.sample, args.method, basis=basis)
        evaluate_registration(args.sample, args.method, overwrite=False)
