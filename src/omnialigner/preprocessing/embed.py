import os
import sys
from typing import Dict

import dask.array as da
import numpy as np
import torch
import torch.nn.functional as F

from omnialigner.preprocessing.attention import get_embeddings_dino, get_embeddings_dino_v2
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageTag, StageSampleTag
from omnialigner.omni_3D import Omni3D
from omnialigner.utils.image_pad import pad_image_to_target
from omnialigner.dtypes import DataType

root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
vendor_path = os.path.join(root_path, 'vendor/istar')
sys.path.append(vendor_path)
from omnialigner.vendor.istar.extract_features import get_embeddings_shift, get_embeddings

def calculated_embed(
        om_data: Omni3D,
        i_layer: int,
        model_name: str="hipt",
        use_dask: bool=False,
        overwrite_cache:bool=False
    ) -> Dict[str, torch.Tensor]: 
    logging.info("Processing embed, stage1, check inputs")
    dict_file_name = StageTag.RAW.get_file_name(om_data.proj_info)
    if dict_file_name is None:
        logging.error("raw files not found, please run `om.read_image()` first.")
        return None


    logging.info("Processing embed, stage2, return cache if exists")
    dict_file_name_embed = StageSampleTag.EMBED.get_file_name(i_layer=i_layer, projInfo=om_data.proj_info)
    if dict_file_name_embed is not None and not overwrite_cache:
        file_name_embed = dict_file_name_embed["embed"]
        logging.info(f"Processing embed, stage2, return cache if exists, {file_name_embed}")
        if os.path.exists(file_name_embed):
            logging.info(f"Processing embed, stage2, cache exists, {file_name_embed}")
            dict_pt = torch.load(file_name_embed)
            return dict_pt

    sample_name = om_data.proj_info.get_sample_name(i_layer=i_layer)
    dtype = om_data.proj_info.get_dtype(i_layer=i_layer)
    is_gray = om_data.tag == DataType.GRAY
    da_raw_hd = om_data.load_tiff(i_layer, zoom_level=0, is_raw=(not is_gray))
    if dtype != "HE":
        da_raw_hd = 255-da_raw_hd[:, :, -1:]
        da_raw_hd = da.repeat(da_raw_hd, 3, axis=2)

    logging.info(f"Processing embed, stage3, read_image_for_embed, sample:{sample_name}, dtype:{dtype}, tag: {om_data.tag}, zoom_level: {om_data.zoom_level}")

    om_data.set_method(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dict_config = om_data.config["pp"]["vit_attention"]["params"]
    kwargs = {
        "tile_size": dict_config["tile_size"],
        "patch_size": dict_config[model_name]["patch_size"],
        "subpatch_size": dict_config[model_name]["subpatch_size"]
    }

    logging.info(f"Processing embed, stage4, get embeddings for {sample_name}: da_affine_hd:{da_raw_hd.shape}, model:{model_name}, VitAttention kwargs: {kwargs}")
    if use_dask:
        out_file = StageSampleTag.EMBED.get_file_name(i_layer, om_data.proj_info, check_exist=False)["embed"]
        tmp_prefix = os.path.dirname(out_file)
        da_emb_cls, da_emb_sub = get_embeddings_dino_v2(
                        da_raw_hd,
                        model_name=model_name,
                        tmp_prefix=f"{tmp_prefix}/{sample_name}_{model_name}",
                        device=device,
                        **kwargs
        )
        out_file = StageSampleTag.EMBED.get_file_name(i_layer, om_data.proj_info, check_exist=False)["embed"]
        if overwrite_cache or not os.path.exists(out_file):
            logging.info(f"Processing embed, stage5, save DASK-embeddings to {out_file}, 'cls':{da_emb_cls.shape}, 'sub':{da_emb_sub.shape}")
            import dask    
            with dask.config.set(scheduler='threads', num_workers=16):
                da.to_zarr(da_emb_cls, f"{tmp_prefix}/{sample_name}_cls_{model_name}.zarr", overwrite=True)
                da.to_zarr(da_emb_sub, f"{tmp_prefix}/{sample_name}_sub_{model_name}.zarr", overwrite=True)

            pt_emb_cls = torch.from_numpy(da_emb_cls.compute())
            pt_emb_sub = torch.from_numpy(da_emb_sub.compute())
            dict_pt = {"cls": pt_emb_cls, "sub": pt_emb_sub}
            torch.save(dict_pt, out_file, _use_new_zipfile_serialization=False)
            return dict_pt
    
    pt_emb_cls, pt_emb_sub = get_embeddings_dino(
                        da_raw_hd.compute(),
                        model_name=model_name,
                        device=device,
                        **kwargs
    )
    out_file = StageSampleTag.EMBED.get_file_name(i_layer, om_data.proj_info, check_exist=False)["embed"]
    dict_pt = {"cls": pt_emb_cls, "sub": pt_emb_sub}
    if overwrite_cache or not os.path.exists(out_file):
        logging.info(f"Processing embed, stage5, save embeddings to {out_file}, 'cls':{dict_pt['cls'].shape}, 'sub':{dict_pt['sub'].shape}")
        torch.save(dict_pt, out_file, _use_new_zipfile_serialization=False)
    
    return dict_pt


def embed(omni: Omni3D, i_layer: int, use_dask=False, overwrite_cache:bool=False) -> Dict[str, torch.Tensor]:
    """Embeds an image layer using a specified model and returns scaled and padded embeddings.

    Args:
        omni (Omni3D): An instance of the Omni3D class containing project and configuration details.
        i_layer (int): The index of the image layer to embed.
        use_dask (bool, optional): Whether to use Dask for distributed computation. Defaults to False.
        overwrite_cache (bool, optional): Whether to overwrite cached embeddings. Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the following keys:
            - "SUB": The subpatch embedding tensor, scaled and padded to match the target image dimensions.
            - "CLS": The class token embedding tensor, scaled and padded to match the target image dimensions.

    Raises:
        ValueError: If required configuration parameters are missing or invalid.
        RuntimeError: If embedding computation fails due to model or data issues.

    Notes:
        - The function uses the `calculated_embed` function to compute embeddings for the specified layer.
        - Embeddings are scaled using the patch and subpatch sizes defined in the configuration.
        - The resulting embeddings are padded to match the dimensions of the original image layer.
    """
    model_name = omni.method
    dict_embed = calculated_embed(omni, i_layer=i_layer, model_name=model_name, use_dask=use_dask, overwrite_cache=overwrite_cache)
    patch_size = omni.config["pp"]["vit_attention"]["params"][omni.method]["patch_size"][0]
    subpatch_size = omni.config["pp"]["vit_attention"]["params"][omni.method]["subpatch_size"][0]
    scale_ratio_cls = patch_size * omni.sizes[omni.zoom_level]
    scale_ratio_sub = patch_size / subpatch_size * omni.sizes[omni.zoom_level]
    img_sub = F.interpolate(dict_embed["sub"].unsqueeze(0), scale_factor=scale_ratio_sub, mode="nearest")
    img_cls = F.interpolate(dict_embed["cls"].permute(2, 0, 1).unsqueeze(0), scale_factor=scale_ratio_cls, mode="nearest")
    img_ = omni.load_tiff(i_layer, zoom_level=omni.zoom_level)
    h, w = img_.shape[:2]
    img_sub = pad_image_to_target(img_sub, (h, w))
    img_cls = pad_image_to_target(img_cls, (h, w))

    return {"SUB": img_sub, "CLS": img_cls}


def __embed_istar_pt(omni: Omni3D, i_layer: int, use_shift=True, device=torch.device("cpu")) -> Dict[str, torch.Tensor]:
    dir_hipt = f"{omni.root_dir}/analysis/{omni.project}/02.dino_feats/{omni.group}/"
    os.makedirs(dir_hipt, exist_ok=True)

    da_wsi = omni.load_tiff(i_layer, zoom_level=0, is_raw=False, resize_to_20x=True)
    sample_name = omni.proj_info.get_sample_name(i_layer=i_layer)
    wsi = da_wsi.compute()
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
    torch.save(dict_pt, f"{dir_hipt}/{sample_name}.pt", _use_new_zipfile_serialization=False)
    return dict_pt
