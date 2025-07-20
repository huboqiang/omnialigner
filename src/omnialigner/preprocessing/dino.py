
import os
import sys
import glob

import cv2
from tqdm import tqdm
import numpy as np
import dask.array as da
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from omnialigner.datasets import generate_WSI
from omnialigner.logging import logger as logging
from omnialigner.utils.image_crop import crop_noise_area

root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
vendor_path = os.path.join(root_path, 'vendor/istar')
sys.path.append(vendor_path)
from omnialigner.vendor.istar.extract_features import get_embeddings_shift, get_embeddings

Tensor_embed_CHW = torch.Tensor
Tensor_mask_HW = torch.Tensor

def calculate_margin_embed(embed: Tensor_embed_CHW, margin: int=16, margin_for_cos: int=5) -> Tensor_embed_CHW:
    back_ground_w = embed[:, :, -(margin+margin_for_cos):-margin].mean([1,2]).view(1, 1, -1)
    back_ground_a = embed[:, margin:(margin+margin_for_cos), :].mean([1,2]).view(1, 1, -1)
    back_ground_s = embed[:, :, margin:(margin+margin_for_cos)].mean([1,2]).view(1, 1, -1)
    back_ground_d = embed[:, -(margin+margin_for_cos):-margin, :].mean([1,2]).view(1, 1, -1)
    back_ground = torch.cat([back_ground_w, back_ground_a, back_ground_s, back_ground_d], axis=0)
    return back_ground.mean(0).view(1,1,-1)



def mask_modify(np_mask):
    
    mask = 1-np_mask
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    label_used, vmax = 1, 0.
    
    for lab in tqdm(range(1, labels.max()+1), desc="select labels"):
        mask_ = (labels==lab)
        score = np.sum(mask_)
        if score > vmax:
            vmax = score
            label_used = lab

    mask_ = (labels==label_used)
    return mask_

def calculate_masks(
        embed: Tensor_embed_CHW, 
        back_ground_embed: torch.Tensor=None, 
        margin: int=16, 
        cos_sim_cutoff: float=0.95, 
        modify_mask: bool=False
    ) -> Tensor_mask_HW:
    """
    calculate masks using dino embedding
    Args:
        embed: Tensor_embed_CHW, dino embedding
        back_ground_embed: torch.Tensor, background embedding
        margin: int, margin for the mask
        cos_sim_cutoff: float, cosine similarity cutoff
        modify_mask: bool, whether to modify the mask
    Returns:
        Tensor_mask_HW, masks. 1 for background, 0 for foreground
    """
    if back_ground_embed is None:
        back_ground_embed = calculate_margin_embed(embed=embed, margin=margin, margin_for_cos=margin//2)

    cos_sim = torch.nn.functional.cosine_similarity(embed.permute(1,2,0), back_ground_embed, dim=2)
    masks = cos_sim > cos_sim_cutoff
    masks[0:margin, :] = 1
    masks[-margin:, :] = 1
    masks[:,0:margin] = 1
    masks[:,-margin:] = 1

    masks = masks.numpy().astype(np.uint8)
    if modify_mask:
        masks = 1-mask_modify(masks)

    return torch.from_numpy(masks)



def apply_masks(embed: Tensor_embed_CHW, masks: Tensor_mask_HW) -> Tensor_embed_CHW:
    pt_embeds = embed.permute(1,2,0)
    pt_embeds[masks] = 0
    return pt_embeds.permute(2,0,1)


def calculate_dino_pt(tiff, generate_WSI, use_shift=True, device=torch.device("cpu")):
    da_arr = generate_WSI(tiff=tiff)
    wsi = da_arr.compute()
    if use_shift:
        emb_cls, emb_sub = get_embeddings_shift(
                        wsi, pretrained=True,
                        device=device)
    else:
        emb_cls, emb_sub = get_embeddings(
                        wsi, pretrained=True,
                        device=device)

    pt_embeds_192 = torch.from_numpy(np.array(emb_cls))
    pt_embeds_384 = torch.from_numpy(np.array(emb_sub))
    dict_pt = {"emb_384": pt_embeds_384, "emb_192": pt_embeds_192}
    return dict_pt



class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

        
    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, max_samples=None, masks_list=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0)

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    
    if fit_pca is None:
        flattened_feats = []
        flattened_masks = []
        for i_feat,feats in enumerate(image_feats_list):
            feats_flatten = flatten(feats, target_size)
            flattened_feats.append(feats_flatten)
            if masks_list is not None and i_feat < len(masks_list):
                mask = masks_list[i_feat].unsqueeze(0).unsqueeze(0).type(torch.float32)
                feats_flatten = flatten(mask, target_size)
                flattened_masks.append(feats_flatten)
        
        x = torch.cat(flattened_feats, dim=0)
        if masks_list is not None:
            mask = torch.cat(flattened_masks, dim=0)[:, 0] > 0.5
            if x.shape[0] != mask.shape[0]:
                logging.warning("x.shape[0] != mask.shape[0]", x.shape, mask.shape)
            else:
                logging.info(f"mask {torch.sum(mask)}/{x.shape[0]} points")
                x = x[~mask]
        

        # Subsample the data if max_samples is set and the number of samples exceeds max_samples
        if max_samples is not None and x.shape[0] > max_samples:
            indices = torch.randperm(x.shape[0])[:max_samples]
            x = x[indices]

        fit_pca = TorchPCA(n_components=dim).fit(x)

    reduced_feats = []
    for i_feat, feats in enumerate(image_feats_list):
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)

        if masks_list is not None and i_feat < len(masks_list):
            mask_ = masks_list[i_feat].ravel()
            x_red_unmask = x_red[~mask_]
            x_red[mask_] = x_red_unmask.min(dim=0, keepdim=True).values

        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca

def plot_feats_lr(l_images, l_feats, fit_pca=None, masks=None, ax=None):
    """
        l_images: list of [H, W, C]
    """
    n_images = len(l_images)
    n_feats = len(l_feats)
    l_feats_pca, pca_model = pca([lr.unsqueeze(0) for lr in l_feats], max_samples=100_000_000, masks_list=masks, fit_pca=fit_pca)
    
    if ax is None:
        fig, ax = plt.subplots(n_feats, 2, figsize=(10, min(300, 5*(n_feats))))
    
    if n_feats == 1:
        i_feat = 0
        if i_feat < n_images:
            ax[0].imshow(l_images[i_feat])
        ax[0].set_title("Image")
    
        ax[1].imshow(l_feats_pca[i_feat][0].permute(1, 2, 0).detach().cpu())
        return l_feats_pca, None
    
    for i_feat in range(n_feats):
        if i_feat < n_images:
            ax[i_feat, 0].imshow(l_images[i_feat])
        
        ax[i_feat, 0].set_title("Image")
    
        ax[i_feat, 1].imshow(l_feats_pca[i_feat][0].permute(1, 2, 0).detach().cpu())
        ax[i_feat, 1].set_title(f"{i_feat}")
        
    return l_feats_pca, pca_model



def group_train_pca(group, project="ANHIR", cos_sim_cutoff=0.95, margin=16, n=1000, ax=None, l_tiffs=None):
    root_dir = "/cluster/home/bqhu_jh/projects/scGaussian3dGen"
    device = torch.device("cpu")
    if l_tiffs is None:
        l_tiffs = glob.glob(f"{root_dir}/analysis/{project}/01.ome_tiff/{group}/*.ome.tif*")
    
    n_feats = min(len(l_tiffs), n)
    savefig = False
    if ax is None:
        savefig = True
        fig, ax = plt.subplots(n_feats, 2, figsize=(10, 5*(n_feats)))
    
    
    l_dasks = []
    l_masks = []
    l_emb192 = []
    l_emb384 = []
    for tiff in tqdm(l_tiffs[0:n_feats]):
        generate_WSI_func = generate_WSI(project)
        da_arr = generate_WSI_func(tiff, zoom_level=-1)
        prefix = os.path.basename(tiff).split(".ome")[0]
        group = tiff.split("/")[-2]
        
        pt_feats = f'{root_dir}/analysis/{project}/02.dino_feats/{group}/{prefix}.pt'
        if not os.path.isfile(pt_feats):
            dict_pt = calculate_dino_pt(tiff, generate_WSI_func, device=device)
        else:
            dict_pt = torch.load(pt_feats)
            dict_pt["emb_384"] = torch.from_numpy(np.array(dict_pt["emb_384"]))
            dict_pt["emb_192"] = torch.from_numpy(np.array(dict_pt["emb_192"]))

        
        masks = calculate_masks(dict_pt["emb_384"], cos_sim_cutoff=cos_sim_cutoff, margin=margin)
        pt_embeds = apply_masks(dict_pt["emb_384"], masks=masks)
        pt_embeds_ = apply_masks(dict_pt["emb_192"], masks=masks)


        l_dasks.append(da_arr)
        l_masks.append(masks)
        l_emb384.append(pt_embeds)
        l_emb192.append(pt_embeds_)

    l_feats_pca, pca_model = plot_feats_lr(l_dasks, l_emb384, masks=l_masks, ax=ax)
    pca_dir = f'{root_dir}/analysis/{project}/03.dino_PCA_feats/{group}/'
    os.makedirs(f"{pca_dir}/{cos_sim_cutoff}", exist_ok=True)
    if savefig:
        feat_pdf = f'{pca_dir}/{group}_{cos_sim_cutoff}.png'
        fig.savefig(feat_pdf)
    
    pt_model_PCA = f'{pca_dir}/{cos_sim_cutoff}/PCA_model.pt'
    torch.save(pca_model, pt_model_PCA, _use_new_zipfile_serialization=False)
    
    for i_feat, feat_pca in enumerate(l_feats_pca):
        tiff = l_tiffs[i_feat]
        prefix = os.path.basename(tiff).split(".ome")[0]
        group = tiff.split("/")[-2]
        
        
        pt_feats_PCA = f'{pca_dir}/{cos_sim_cutoff}/{prefix}.PCA.pt'
        
        torch.save(feat_pca, pt_feats_PCA, _use_new_zipfile_serialization=False)
    

def group_apply_pca(tiff, fit_pca, project="ANHIR", cos_sim_cutoff=0.95, margin=16, ax=None):
    n_feats = 1
    root_dir = "/cluster/home/bqhu_jh/projects/scGaussian3dGen"

    savefig = False
    if ax is None:
        savefig = True
        fig, ax = plt.subplots(n_feats, 2, figsize=(10, 5*(n_feats)))
    
    
    l_dasks = []
    l_masks = []
    l_emb192 = []
    l_emb384 = []
    
    generate_WSI_func = generate_WSI(project)
    da_arr = generate_WSI_func(tiff, zoom_level=-1)
    prefix = os.path.basename(tiff).split(".ome")[0]
    group = tiff.split("/")[-2]
    
    pt_feats = f'{root_dir}/analysis/{project}/02.dino_feats/{group}/{prefix}.pt'
    if not os.path.isfile(pt_feats):
        device = torch.device("cpu")
        dict_pt = calculate_dino_pt(tiff, generate_WSI_func, device=device)
    else:
        dict_pt = torch.load(pt_feats)
        dict_pt["emb_384"] = torch.from_numpy(np.array(dict_pt["emb_384"]))
        dict_pt["emb_192"] = torch.from_numpy(np.array(dict_pt["emb_192"]))

    
    masks = calculate_masks(dict_pt["emb_384"], cos_sim_cutoff=cos_sim_cutoff, margin=margin)
    pt_masks = f'{root_dir}/analysis/{project}/02.dino_feats/{group}/{prefix}.masks.png'
    cv2.imwrite(pt_masks, 255*masks.numpy().astype(np.uint8))
    pt_embeds = apply_masks(dict_pt["emb_384"], masks=masks)
    pt_embeds_ = apply_masks(dict_pt["emb_192"], masks=masks)


    l_dasks.append(da_arr)
    l_masks.append(masks)
    l_emb384.append(pt_embeds)
    l_emb192.append(pt_embeds_)

    l_feats_pca, _ = plot_feats_lr(l_dasks, l_emb384, fit_pca=fit_pca, masks=l_masks, ax=ax)
    pca_dir = f'{root_dir}/analysis/{project}/03.dino_PCA_feats/{group}/'
    os.makedirs(pca_dir, exist_ok=True)
    
    if savefig:
        feat_pdf = f'{pca_dir}/{group}_{cos_sim_cutoff}.png'
        fig.savefig(feat_pdf)
    
    
    for _, feat_pca in enumerate(l_feats_pca):
        prefix = os.path.basename(tiff).split(".ome")[0]
        group = tiff.split("/")[-2]
        
        os.makedirs(f"{pca_dir}/{cos_sim_cutoff}", exist_ok=True)
        pt_feats_PCA = f'{pca_dir}/{cos_sim_cutoff}/{prefix}.PCA.pt'    
        torch.save(feat_pca, pt_feats_PCA, _use_new_zipfile_serialization=False)

def calculate_crop_from_mask(
            np_mask: np.ndarray,
            threshold: float=0.1,
            margin_extend: float=0.1,
            min_length: int=100
    ):
    H, W = np_mask.shape[0], np_mask.shape[1]
    img, coords1 = crop_noise_area(np_mask, threshold=threshold, min_length=min_length)
    img, coords2 = crop_noise_area(img, threshold=threshold, min_length=min_length)
    np_coords = coords1 + coords2
    np_coords = np.array([
          np_coords[0]/H,
          np_coords[1]/W,
        1-np_coords[2]/H,
        1-np_coords[3]/W
    ])

    np_coords[0] = max(0.0, np_coords[0]-margin_extend)
    np_coords[1] = max(0.0, np_coords[1]-margin_extend)
    np_coords[2] = min(1.0, np_coords[2]+margin_extend)
    np_coords[3] = min(1.0, np_coords[3]+margin_extend)
    return (255*img).astype(np.uint8), np_coords


def calculate_crop_margin_dino(
        da_arr: da.Array,
        margin_extend: float=0.1,
        cos_sim_cutoff: float=0.9,
        threshold: float=0.1,
        min_length: int=100,
        margin: int=16,
        device=torch.device("cpu")
    ):
    """
    Calculate masks for cropping an image using DINO embeddings.

    Args:
        da_arr (da.Array): Input image as a Dask array.
        margin_extend (float, optional): Margin extension factor for cropping. Defaults to 0.1.
        cos_sim_cutoff (float, optional): Cosine similarity cutoff for mask calculation. Defaults to 0.9.
        threshold (float, optional): Threshold for noise area cropping. Defaults to 0.1.
        min_length (int, optional): Minimum length for cropping regions. Defaults to 100.
        margin (int, optional): Margin size for mask calculation. Defaults to 16.
        device (torch.device, optional): Device to use for computation (e.g., CPU or GPU). Defaults to CPU.

    Returns:
        Tuple[np.ndarray, np.ndarray, Any]: 
            - np_mask (np.ndarray): Binary mask indicating the cropped region.
            - np_coords (np.ndarray): Normalized coordinates of the cropped region [top, left, bottom, right].
            - emb_sub (Any): Subpatch embeddings generated by the DINO model.

    """
    _, emb_sub = get_embeddings(
                            da_arr.compute(), pretrained=True,
                            device=device)

    pt_embeds_384 = torch.from_numpy(np.array(emb_sub))
    
    masks = calculate_masks(pt_embeds_384, margin=margin, cos_sim_cutoff=cos_sim_cutoff)
    np_mask = (1-masks.float()).detach().cpu().numpy()
    
    np_mask_cropped, np_coords = calculate_crop_from_mask(np_mask.copy(), threshold, margin_extend, min_length)
    return np_mask, np_coords, emb_sub
