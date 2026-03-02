"""
Common utilities for 3D alignment methods
"""
import os
from typing import Tuple, List
import random
import numpy as np
import anndata as ad
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import KDTree
import glob

import omnialigner as om
from omnialigner.plotting.h5ad_viz import keypoints_gpd, gdf_shape_to_image
plt = om.pl.plt

def refine_raw_with_ref(adata, i_layer:int=0, src_obsm_key:str="spatial_3D_center", tar_obsm_key:str="spatial_ref"):
    l_layers = adata.obs["i_layer"].unique().tolist()
    adata_0 = adata[adata.obs["i_layer"] == l_layers[i_layer]]
    kpt_src = adata_0.obsm[src_obsm_key][:, 0:2]
    kpt_tar = adata_0.obsm[tar_obsm_key]
    kpt_src_ = np.hstack( (kpt_src, np.ones([kpt_src.shape[0], 1])) )
    kpt_tar_ = np.hstack( (kpt_tar, np.ones([kpt_tar.shape[0], 1])) )
    mat_moved, _, _, _ = np.linalg.lstsq(kpt_src_, kpt_tar_)

    kpt_src_all = adata.obsm[src_obsm_key][:, 0:2]
    kpt_src_all_ = np.hstack( (kpt_src_all, np.ones([kpt_src_all.shape[0], 1])) )
    kpt_moved_all = np.dot(kpt_src_all_, mat_moved)[:, 0:2]
    
    kpt_spatial_3d_moved = np.hstack( (kpt_moved_all, adata.obsm[src_obsm_key][:, 2:3]) )
    return kpt_spatial_3d_moved


def inspect_reduced_coords(adata):
    l_layers = adata.obs["i_layer"].unique().tolist()
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(1, 3, 1)
    for i_layer in l_layers:
        adata_z = adata[adata.obs["i_layer"] == i_layer]
        kpt = adata_z.obsm["spatial_3D_center"]
        ax1.scatter(kpt[:,0], kpt[:,1], s=6)
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(0, 1280)
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(1, 3, 2)
    for i_layer in l_layers:
        adata_z = adata[adata.obs["i_layer"] == i_layer]
        kpt = adata_z.obsm["spatial_ref"]
        ax2.scatter(kpt[:,0], kpt[:,1], s=6)
    ax2.set_xlim(0, 1280)
    ax2.set_ylim(0, 1280)
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(1, 3, 3)
    for i_layer in l_layers:
        adata_z = adata[adata.obs["i_layer"] == i_layer]
        kpt = adata_z.obsm["spatial_3D"]
        ax3.scatter(kpt[:,0], kpt[:,1], s=6)
    ax3.set_xlim(0, 1280)
    ax3.set_ylim(0, 1280)
    ax3.invert_yaxis()
    return fig

def assign_undef_label_knn(adata, col_label="spatial_domain", basis="spatial", label_undef="Unknown", k=5):
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[basis]
    labels = adata.obs[col_label].tolist()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    assigned_labels = []
    for i in range(coords.shape[0]):
        if labels[i] != label_undef:
            assigned_labels.append(labels[i])
        else:
            neighbor_labels = [labels[idx] for idx in indices[i] if labels[idx] != label_undef]
            if neighbor_labels:
                most_common = max(set(neighbor_labels), key=neighbor_labels.count)
                assigned_labels.append(most_common)
            else:
                assigned_labels.append(label_undef)
                print(f"Warning: Cell {i} could not be assigned a label.")
    
    return assigned_labels

def scale_coords(np_coords, np_min=None, np_max=None, extent=0.1):
    np_min = np_coords.min(0) if np_min is None else np_min
    np_max = np_coords.max(0) if np_max is None else np_max
    np_scaled = (np_coords - np_min) / (np_max - np_min)
    np_scaled = (np_scaled+extent) / (1 + extent*2)
    return np_scaled


def centerize_coords(np_coords, canvas_size=5000):
    np_min = np_coords.min(0)
    np_max = np_coords.max(0)
    np_size = np_max - np_min
    offset = (canvas_size - np_size) / 2
    np_centered = np_coords - np_min + offset
    return np_centered / canvas_size


def get_sample_config(sample:str):
    """Get sample configuration from command line or default"""

    config_dict = {}
    if len(sample) > 4 and sample[0:4] == "abca":
        dict_abca_dict = {
            "abca-1": "zhuang_ABCA_1",
            "abca-2": "zhuang_ABCA_2",
            "abca-3": "zhuang_ABCA_3",
            "abca-4": "zhuang_ABCA_4",
        }
        h5ad_path = f"/cluster/home/bqhu_jh/projects/bqhu_jh/DECIPHER/experiments/mouse_brain_MERFISH/data/zhuang_dataset/{sample}"
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/zhuang/{dict_abca_dict[sample]}/"
        ref_layer = 4 if sample == "abca-2" else 0
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "parcellation_structure",
            "ref_layer": ref_layer,
            "radius": (6, 6)
        }
        # return sample, h5ad_path, out_root, "mapped_celltype", (6, 6)

    if sample == "mouse_E11.5_embryo":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/spateo/{sample}"
        h5ad_path = f"{out_root}/{sample}.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "mapped_celltype",
            "radius": (6, 6)
        }

    if sample == "mouse_E9.5_embryo":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/spateo/{sample}"
        h5ad_path = f"{out_root}/{sample}.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "mapped_celltype",
            "radius": (6, 6)
        }

    if sample == "slideseq_brain":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/slideseq/{sample}"
        h5ad_path = f"{out_root}/../All_Pucks_h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "CCF_Name",
            "radius": (6, 6)
        }
        # return sample, h5ad_path, out_root, "parcellation_structure", (8, 8)
    
    if sample == "ST_mouse_brain":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/SPACEL/{sample}"
        h5ad_path = f"{out_root}/mouse_brain_st.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "spatial_domain",
            "ref_layer": 36,
            "radius":  (28, 24)
        }
    
    if sample == "stereo_seq_mouse_embryo":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/SPACEL/{sample}"
        h5ad_path = f"{out_root}/mouse_embryo_all_slices.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "spatial_domain",
            "radius":  (6, 6)
        }
    
    if sample == "merfish_mouse_brain":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/SPACEL/{sample}"
        h5ad_path = f"{out_root}/merfish_mouse_brain_SPACEL_aligned.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "spatial_domain",
            "radius":  (6, 6),
            "ref_layer": 20,
            "refine_pose": False,
        }
        
    if sample == "starmap_3d_mouse_brain":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/SPACEL/{sample}"
        h5ad_path = f"{out_root}/starmap_3d_mouse_brain.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "spatial_domain",
            "radius":  (20, 20)
        }
    
    if sample == "SmallIntestine":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/Loki/{sample}"
        h5ad_path = f"{out_root}/SmallIntestine.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "spatial_domain",
            "pca_feature": "txt_features",
            "radius":  (6, 6),
            "refine_pose": True,
            "refine_pose_axes": [1, 2, 3, 4],
        }

    if sample == "metastatic_lymph_node":
        out_root = f"/cluster/home/bqhu_jh/projects/bqhu_jh/isoST/data/openST/{sample}"
        h5ad_path = f"{out_root}/GSE251926_metastatic_lymph_node_3d.h5ad"
        config_dict = {
            "sample": sample,
            "h5ad_path": h5ad_path,
            "out_root": out_root,
            "label_col": "annotation",
            "radius":  (6, 6),
            "refine_pose": True,
            "refine_pose_axes": [1, 2, 3, 4],
        }
        # return sample, h5ad_path, out_root, "spatial_domain", (28, 24)
    
    config_dict["h"] = 1280
    config_dict["w"] = 1280
    return config_dict

def random_sample(coords_t, nodenum, seed_t=2):
    """Randomly sample coordinates"""
    random.seed(seed_t)
    if coords_t.shape[0] <= nodenum:
        return np.arange(coords_t.shape[0])
    
    sub_node_idx = np.sort(random.sample(range(coords_t.shape[0]), nodenum))
    return sub_node_idx



def process_coords_abca(np_spatial_raw, sample):
    """Process spatial coordinates for different samples"""
    np_spatial = np_spatial_raw.copy()
    np_spatial = np_spatial[:, 0:2] / 8.0
    if sample == "abca-3" or sample == "abca-4":
        np_spatial = (np_spatial_raw + 1.25) / 16
    
    return np_spatial * 25600.

def process_coords_default(np_spatial_raw, canvas_size=5000):
    centered_coords = centerize_coords(np_spatial_raw, canvas_size=canvas_size)
    return centered_coords * canvas_size


def extract_ref_coords(adata, sample):
    if sample == "mouse_E9.5_embryo":
        return adata.obsm["3d_align_spatial"][:, 0:2] / 4200.

    if sample == "mouse_E11.5_embryo":
        return (adata.obsm["z_correction"][:, 0:2] + np.array([0, 400])) / 2_400.

    if sample == "slideseq_brain":
        return (adata.obsm["spatial_aligned"][:, 0:2] + np.array([-120, -50])) / 240.

    if sample == "ST_mouse_brain":
        return (adata.obsm["spatial_aligned"].values[:, 0:2] + 4_000) / 12_000.

    if sample == "stereo_seq_mouse_embryo":
        return (adata.obsm["spatial_aligned"].values[:, 0:2] + np.array([350, 320])) / 600.
    
    if sample == "merfish_mouse_brain":
        return (adata.obsm["spatial_aligned"].values + np.array([1000, 2500])) / 4000.

    if sample == "starmap_3d_mouse_brain":
        return (adata.obsm["spatial_truth"] + np.array([100, 250])) / 2000.

    if sample == "SmallIntestine":
        return (adata.obsm["spatial_aligned"] + np.array([300, 100])) / 2400.

    if sample == "metastatic_lymph_node":
        return (adata.obsm["spatial_3d_aligned"][:, 0:2] + np.array([500, 4000])) / 18000

    if sample.split("-")[0] == "abca":
        cell_pos_ref = adata.obs[["z_ccf", "y_ccf"]].values  / 8.0
        cell_pos_ref = cell_pos_ref * np.array([1.2, 1.0])

        if sample == "abca-3" or sample == "abca-4":
            cell_pos_ref = (adata.obs[["x_ccf", "y_ccf"]].values + 1.25) / 16 + np.array([0.0, 0.2])

        return cell_pos_ref
    
    return None

def extract_raw_coords_kpts(kpt, sample):
    if sample == "mouse_E11.5_embryo":
        return process_coords_default(kpt, canvas_size=25000.) / 25000.
    
    if sample == "mouse_E9.5_embryo":
        return process_coords_default(kpt, canvas_size=5000.) / 5000.

    if sample == "slideseq_brain":
        return process_coords_default(kpt, canvas_size=6_000) / 6_000.

    if sample == "ST_mouse_brain":
        return process_coords_default(kpt, canvas_size=10_000) / 10_000.

    if sample == "stereo_seq_mouse_embryo":
        return process_coords_default(kpt, canvas_size=1_000) / 1_000.

    if sample.split("-")[0] == "abca":
        return process_coords_abca(kpt, sample) / 25600.

    if sample == "merfish_mouse_brain":
        return process_coords_default(kpt, canvas_size=4_000) / 4_000.
    
    if sample == "starmap_3d_mouse_brain":
        return process_coords_default(kpt, canvas_size=4_000) / 4_000.

    if sample == "SmallIntestine":
        return process_coords_default(kpt+ np.array([300, 300]), canvas_size=2_400) / 2_400.

    if sample == "metastatic_lymph_node":
        return process_coords_default(kpt+ np.array([1500, 1500]), canvas_size=18_000) / 18_000.

    np_min = kpt.min(0)
    np_max = kpt.max()
    np_scaled = process_coords_default(kpt+ np_min, canvas_size=np_max) / np_max
    return np_scaled

def extract_raw_coords(adata, sample):
    return extract_raw_coords_kpts(adata.obsm["spatial"], sample)

def load_or_create_adata(h5ad_path, out_root, sample, h:int=1280, w:int=1280, ref_layer:int=0, **kwargs):
    """Load preprocessed data or create if not exists"""
    from preprocess_for_isoST import build_zhuang_merfish_h5ad
    
    reduced_path = os.path.join(out_root, 'spateo', f'reduced.h5ad')
    
    if not os.path.exists(reduced_path):
        import dynamo as dyn
        adata: ad.AnnData = None
        if len(sample) > 3 and sample[0:4] == "abca":
            adata = build_zhuang_merfish_h5ad(data_dir=h5ad_path)
            adata.obs['i_layer'] = [int(x.split(".")[-1]) for x in adata.obs['brain_section_label']]
            np_3d = np.hstack([
                # process_coords_abca(adata.obsm["spatial"][:, 0:2], sample),
                extract_raw_coords_kpts(adata.obsm["spatial"][:, 0:2], sample),
                adata.obs["i_layer"].values[:, None]
            ])
            

        if sample == "mouse_E11.5_embryo":
            adata_raw = ad.read_h5ad(h5ad_path)
            adata = adata_raw[adata_raw.obsm["spatial"][:,0] > 3000]
            adata.obs["i_layer"] = [ int(x[1:]) for x in adata.obs["slice_id"] ]
            l_layers = sorted(adata.obs["i_layer"].unique().tolist())
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = adata[adata.obs["i_layer"] == i_layer]
                # adata_sub.obsm["spatial"] = extract_raw_coords(adata_sub, sample)
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"][:, 0:2], sample)
                l_adatas.append(adata_sub)

            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                            adata.obsm["spatial"],
                            adata.obs["i_layer"].values[:, None]
            ])

        if sample == "mouse_E9.5_embryo":
            adata = ad.read_h5ad(h5ad_path)
            adata.obs["i_layer"] = (adata.obsm["3d_align_spatial"][:, 2] // 20).astype(np.int32)
            l_layers = sorted(adata.obs["i_layer"].unique().tolist())
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = adata[adata.obs["i_layer"] == i_layer]
                # adata_sub.obsm["spatial"] = extract_raw_coords(adata_sub, sample)
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"][:, 0:2], sample)
                l_adatas.append(adata_sub)

            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                            adata.obsm["spatial"],
                            adata.obs["i_layer"].values[:, None]
            ])

        if sample == "slideseq_brain":
            h5ad_files = glob.glob(f"{h5ad_path}/*.h5ad")
            l_layers = sorted([ int(x[:-5].split("_")[-1]) for x in h5ad_files])
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = ad.read_h5ad(h5ad_files[idx])
                adata_sub.obs["i_layer"] = int(i_layer)
                if "CCF_Name" not in adata_sub.obs.columns:
                    print(f"Warning: CCF_Name not in obs columns of {sample} for layer {i_layer}. Skip")
                    continue

                if "CCF_Z" not in adata_sub.obs.columns or "CCF_Y" not in adata_sub.obs.columns:
                    print(f"Warning: CCF_X/Y/Z not in obs columns of {sample} for layer {i_layer}. Skip")
                    continue

                adata_sub = adata_sub[(adata_sub.obs["CCF_Name"] != "NA") & (adata_sub.obs["IsOutsideCCF"] == False)]
                adata_sub.obsm["spatial"] = adata_sub.obs[["Raw_Slideseq_X", "Raw_Slideseq_Y"]].values
                adata_sub.obsm["spatial_aligned"] = adata_sub.obs[["CCF_Z", "CCF_Y"]].values
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"], sample)
                l_adatas.append(adata_sub)
            
            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                # process_coords_default(adata_sub.obsm["spatial"], canvas_size=10_000),
                adata.obsm["spatial"],
                adata.obs["i_layer"].values[:, None]
            ])

        if sample == "ST_mouse_brain":
            adata = ad.read_h5ad(h5ad_path)
            adata.obs["i_layer"] = (adata.obs["slice"].values).astype(np.int32)
            np_3d = np.hstack([
                # process_coords_default(adata.obsm["spatial"], canvas_size=10_000),
                extract_raw_coords_kpts(adata.obsm["spatial"], sample),
                adata.obs["i_layer"].values[:, None]
            ])
        
        if sample == "stereo_seq_mouse_embryo":

            adata = ad.read_h5ad(h5ad_path)
            adata.obs["i_layer"] = (adata.obs["slice"].values).astype(np.int32)
            l_layers = sorted(adata.obs["i_layer"].unique().tolist())
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = adata[adata.obs["i_layer"] == i_layer]
                # adata_sub.obsm["spatial"] = extract_raw_coords(adata_sub, sample)
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"][:, 0:2], sample)
                l_adatas.append(adata_sub)

            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                # process_coords_default(adata.obsm["spatial"], canvas_size=1_000),
                adata.obsm["spatial"],
                adata.obs["i_layer"].values[:, None]
            ])
        
        if sample == "merfish_mouse_brain":
            adata_ = ad.read_h5ad(h5ad_path)
            adata_.obs["i_layer"] = np.array([ x.split("_slice")[-1] for x in adata_.obs["slice_id"].values]).astype(np.int32)
            l_layers = adata_.obs["slice_id"].drop_duplicates().tolist()
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = adata_[adata_.obs["slice_id"] == i_layer]
                adata_sub.obs["i_layer"] = np.array([ x.split("_slice")[-1] for x in adata_sub.obs["slice_id"].values]).astype(np.int32)
                # adata_sub.obsm["spatial"] = process_coords_default(adata_sub.obsm["spatial"].values, canvas_size=4_000)
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"].values, sample)
                l_adatas.append(adata_sub)

            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                adata.obsm["spatial"],
                adata.obs["i_layer"].values[:, None]
            ])
        
        if sample == "starmap_3d_mouse_brain":
            adata_ = ad.read_h5ad(h5ad_path)
            adata_.obs["i_layer"] = np.array([ x for x in adata_.obs["slice"].values]).astype(np.int32)
            l_layers = adata_.obs["slice"].drop_duplicates().tolist()
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = adata_[adata_.obs["slice"] == i_layer]
                # adata_sub.obsm["spatial"] = process_coords_default(adata_sub.obsm["spatial"].values, canvas_size=4_000)
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"].values, sample)
                l_adatas.append(adata_sub)

            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                adata.obsm["spatial"],
                adata.obs["i_layer"].values[:, None]
            ])

        if sample == "SmallIntestine":
            ## loki.preprocess.prepare_data_for_alignment(ad_path)
            ##     See /cluster/home/bqhu_jh/projects/bqhu_jh/isoST/script/mouse_brain/make_Loki_adata.py
            adata = ad.read_h5ad(h5ad_path)
            adata.obsm["spatial"] = extract_raw_coords_kpts(adata.obsm["spatial"], sample)
            np_3d = np.hstack([
                adata.obsm["spatial"],
                adata.obs["i_layer"].values[:, None]
            ])

        if sample == "metastatic_lymph_node":
            adata_ = ad.read_h5ad(h5ad_path)
            # adata_.obs["i_layer"] = np.array([ x for x in adata_.obs["slice"].values]).astype(np.int32)
            adata_.obs['i_layer'] = adata_.obs['n_section'].astype(int)
            l_layers = sorted(adata_.obs["n_section"].unique())
            l_adatas = []
            for idx,i_layer in enumerate(l_layers):
                adata_sub = adata_[adata_.obs["i_layer"] == i_layer]
                adata_sub.obsm["spatial"] = extract_raw_coords_kpts(adata_sub.obsm["spatial"].toarray(), sample)
                l_adatas.append(adata_sub)

            adata = ad.concat(l_adatas)
            np_3d = np.hstack([
                adata.obsm["spatial"],
                adata.obs["i_layer"].values[:, None]
            ])
            dyn.tools.clustering.calc_sz_factor(adata)
            adata.obs["rawSize_Factor"] = adata.obs["raw_Size_Factor"]

        extend_ratio = kwargs.get("extend_ratio", 0.2)
        np_3d = np_3d * np.array([w, h, 1])
        np_ref_ = extract_ref_coords(adata, sample) * np.array([w, h])
        np_ref = scale_coords(np_ref_, np_max=np.array([w, h]), np_min=np.array([0, 0]), extent=extend_ratio) * np.array([w, h])
        adata.obsm["spatial_ref"] = np_ref
        adata.obsm['spatial_3D_center'] = np_3d
        adata.obsm['spatial_3D'] = refine_raw_with_ref(adata, i_layer=ref_layer, src_obsm_key="spatial_3D_center", tar_obsm_key="spatial_ref")
        fig = inspect_reduced_coords(adata)
        fig.savefig(f"{out_root}/overview_{sample}.png")
        preprocessor = dyn.preprocessing.Preprocessor()
        preprocessor.preprocess_adata(adata, recipe="monocle")
        dyn.tl.reduceDimension(adata, basis="pca")
        os.makedirs(os.path.join(out_root, 'spateo'), exist_ok=True)
        adata.obs = adata.obs.drop(['ntr'], axis=1) if 'ntr' in adata.obs.columns else adata.obs
        adata.var = adata.var.drop(['ntr'], axis=1) if 'ntr' in adata.var.columns else adata.var
        adata.write_h5ad(reduced_path)
        return adata

    adata = ad.read_h5ad(reduced_path)
    return adata


def prepare_slices(adata):
    """Prepare slices from adata"""
    z_height = np.unique(adata.obsm['spatial_3D'][:, 2])
    adata.obsm["spatial_2D"] = adata.obsm['spatial_3D'][:, 0:2].copy()
    slides_raw = [adata[adata.obsm['spatial_3D'][:, 2] == z] for z in z_height]
    coords_t = {int(z): adata[adata.obsm['spatial_3D'][:, 2] == z].obsm["spatial_3D"][:, 0:2] for z in z_height}
    
    return slides_raw, coords_t, z_height

def kpt_to_png(np_kpts, l_colors, l_clusters=None, h=1280, w=1280, using_color=True, radius:Tuple=(6,6)):
    if not using_color:
        gdf = keypoints_gpd(np_kpts, key={"clusters": l_clusters}, color_key={"color": l_colors}, radius=radius[0])#.plot()
        return gdf_shape_to_image(gdf, key="clusters", w=w, h=h)[:, :, np.newaxis]

    gdf = keypoints_gpd(np_kpts, key={"clusters": l_clusters}, color_key={"color": l_colors}, radius=radius[1])#.plot()
    np_uint8 = gdf_shape_to_image(gdf, key="clusters", color=gdf["color"], w=w, h=h)
    return np_uint8


def assign_kpt_interpolate(
    adata_sub: ad.AnnData,
    adata_all: ad.AnnData,
    key_common: str = "spatial",
    key_map: str = "align_spatial",
    method: str = "linear",
    k:int = 5
) -> ad.AnnData:
    """
    Assign align_spatial coordinates to adata_all using spatial interpolation from adata_sub
    based on relative neighborhood relationships.
    
    Parameters:
        adata_sub: Subsampled AnnData object with:
            - obsm['spatial']: Original spatial coordinates (n_sub, dim)
            - obsm['align_spatial']: Aligned spatial coordinates (n_sub, dim)
        adata_all: Full AnnData object with:
            - obsm['spatial']: Original spatial coordinates (n_all, dim)
        method: Interpolation method ('linear' or 'nearest'):
            - 'linear': Linear interpolation using nearest neighbors
            - 'nearest': Nearest neighbor interpolation
        k: Number of neighbors for linear interpolation (ignored for 'nearest'). 
           Must be ≥ dim (2 for 2D data, 3 for 3D data).
    
    Raises:
        ValueError: If required obsm keys are missing, dimensions mismatch, or invalid parameters
    """
    # Input validation
    
    for key in [key_common, key_map]:
        if key not in adata_sub.obsm:
            raise ValueError(f"adata_sub missing required obsm key: {key}")
    
    for key in [key_common]:
        if key not in adata_all.obsm:
            raise ValueError(f"adata_all missing required obsm key: {key}")
    
    if adata_sub.shape[0] == adata_all.shape[0]:
        return adata_sub

    # Validate interpolation parameters
    if method not in ["linear", "nearest"]:
        raise ValueError(f"Invalid method: {method}. Must be 'linear' or 'nearest'")
    
    if method == "linear":
        if k > len(adata_sub):
            raise ValueError(f"k ({k}) cannot exceed number of subsampled points ({len(adata_sub)})")
    
    # Extract coordinates and sample names
    sub_spatial = adata_sub.obsm[key_common][:, 0:2].copy()
    sub_aligned = adata_sub.obsm[key_map][:, 0:2].copy()
    all_spatial = adata_all.obsm[key_common][:, 0:2].copy()
    sub_names = set(adata_sub.obs_names)
    all_names = adata_all.obs_names
    
    # Initialize aligned coordinates array
    all_aligned = np.zeros_like(all_spatial)
    
    # 1. Handle subsampled points (exact match) first
    for i, name in enumerate(all_names):
        if name in sub_names:
            idx = adata_sub.obs_names.get_loc(name)
            all_aligned[i] = sub_aligned[idx]
    
    # 2. Get non-sub sampled points for interpolation
    non_sub_mask = ~np.isin(all_names, list(sub_names))
    non_sub_spatial = all_spatial[non_sub_mask]
    
    if len(non_sub_spatial) == 0:
        adata_all.obsm[key_map] = all_aligned
        print(f"All points are in subsample - direct assignment complete (shape: {all_aligned.shape})")
        return
    
    # 3. Build interpolator with SciPy
    if method == "nearest":
        # Nearest neighbor interpolation
        interpolator = NearestNDInterpolator(sub_spatial, sub_aligned)
        non_sub_aligned = interpolator(non_sub_spatial)
    
    else:  # linear interpolation
        # Use KDTree to get k-nearest neighbors for stable linear interpolation
        kd_tree = KDTree(sub_spatial)
        non_sub_aligned = np.zeros((len(non_sub_spatial), 2))
        for i, point in enumerate(non_sub_spatial):
            # Get k nearest neighbors with adaptive expansion for degenerate cases
            k_current = k
            max_attempts = 3
            attempts = 0
            
            while attempts < max_attempts:
                distances, indices = kd_tree.query(point, k=min(k_current, len(sub_spatial)))
                neighbor_spatial = sub_spatial[indices]
                neighbor_aligned = sub_aligned[indices]
                
                # Check if neighbors are degenerate (all same x or y coordinate)
                x_unique = np.unique(neighbor_spatial[:, 0]).size > 1
                y_unique = np.unique(neighbor_spatial[:, 1]).size > 1
                
                if x_unique and y_unique:
                    # Sufficient diversity for triangulation
                    break
                
                # Expand k and retry
                attempts += 1
                k_current += 5
                
                if k_current > len(sub_spatial):
                    raise ValueError(
                        f"Point {i}: Cannot find non-degenerate neighbors after {max_attempts} attempts. "
                        f"All {len(sub_spatial)} points may be collinear."
                    )
            
            # Linear interpolation on local neighbors
            local_interp = LinearNDInterpolator(neighbor_spatial, neighbor_aligned)
            # Get interpolated value (fallback to nearest if singular)
            interp_val = local_interp(point)
            if np.any(np.isnan(interp_val)):
                interp_val = neighbor_aligned[np.argmin(distances)]
            non_sub_aligned[i] = interp_val
    
    # Assign interpolated values back to full array
    all_aligned[non_sub_mask] = non_sub_aligned
    
    # Final assignment to AnnData object
    adata_all.obsm[key_map] = all_aligned
    return adata_all

def adata_feature_to_rgb(adata:ad.AnnData, feature_key:str="txt_features", layer_col:str='i_layer', ref_layers:List[int]=None, rgb_key:str='feature_rgb') -> ad.AnnData:
    """Convert feature matrix to RGB colors and store in adata.obs.

    Args:
        adata: AnnData object containing the feature matrix.
        feature_key: Key in adata.obsm where the feature matrix is stored.
        laber_col: Column in adata.obs indicating layer information.
        ref_layers: List of layer values to use as reference for PCA fitting.
        rgb_key: Key in adata.obs where the RGB colors will be stored.
    Returns:
        Updated AnnData object with RGB colors stored in adata.obs.
    """
    from sklearn.decomposition import PCA
    tar_features = adata[adata.obs[layer_col].isin(ref_layers)].obsm[feature_key]
    src_features = adata.obsm[feature_key]

    pca = PCA(n_components=3)
    pca_fit_tar = pca.fit(tar_features)
    features = pca_fit_tar.transform(src_features)  # Transform target features

    features_min = features.min(axis=0)
    features_max = features.max(axis=0)
    features_norm = (features - features_min) / (features_max - features_min + 1e-8)
    features_norm[:,[0,1,2]] = features_norm[:,[0,2,1]]
    adata.obsm[rgb_key] = features_norm
    return adata

