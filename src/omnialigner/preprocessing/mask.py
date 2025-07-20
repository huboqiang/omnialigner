import os
import numpy as np

from omnialigner.dtypes import Np_image_Mask
from omnialigner.logging import logger as logging
from omnialigner.cache_files import StageSampleTag
from omnialigner.omni_3D import Omni3D
from omnialigner.preprocessing.embed import embed
from omnialigner.preprocessing.dino import calculate_masks

def mask(omni: Omni3D, i_layer: int, overwrite_cache:bool=False, **kwargs_tag) -> Np_image_Mask:
    """
    Calculate the mask for the given layer using ViT embeddings.

    Args:
        omni (Omni3D): The Omni3D object.
        i_layer (int): The layer index.
        overwrite_cache (bool): Whether to overwrite the cache.
        **kwargs_tag: Additional keyword arguments for the mask calculation.

    Returns:
        Np_image_Mask: The mask for the given layer.
    """
    logging.info(f"Processing mask, stage1, check inputs")

    cos_sim_cutoff = kwargs_tag.get("cos_sim_cutoff", 0.4)
    margin = kwargs_tag.get("margin", 16)
    modify_mask = kwargs_tag.get("modify_mask", True)
    tag_used = kwargs_tag.get("tag_used", "SUB")

    dict_file_name = StageSampleTag.EMBED.get_file_name(i_layer=i_layer, projInfo=omni.proj_info)
    if dict_file_name is None:
        logging.error(f"embed files not found, please run `om.pp.embed()` first.")
        return None
    
    logging.info(f"Processing mask, stage2, return cache if exists")
    dict_file_name = StageSampleTag.MASK.get_file_name(i_layer=i_layer, projInfo=omni.proj_info)
    if dict_file_name is not None and not overwrite_cache:
        file_name_mask = dict_file_name["mask"]
        logging.info(f"Processing mask, stage2, return cache if exists, {file_name_mask}")
        if os.path.exists(file_name_mask):
            logging.info(f"Processing mask, stage2, cache exists, {file_name_mask}")
            np_masks = np.load(file_name_mask)
            return np_masks
    
    logging.info(f"Processing mask, stage3, calculate masks")
    dict_embed = embed(omni, i_layer, overwrite_cache=False) # too expensive to overwrite cache for ViT
    tensor_emb = dict_embed[tag_used].squeeze(0)
    np_masks = calculate_masks(tensor_emb, cos_sim_cutoff=cos_sim_cutoff, modify_mask=modify_mask, margin=margin)

    out_file = StageSampleTag.MASK.get_file_name(i_layer=i_layer, projInfo=omni.proj_info, check_exist=False)["mask"]
    if overwrite_cache or not os.path.exists(out_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        logging.info(f"Processing mask, stage4, save masks to {out_file}, 'masks':{np_masks.shape}")
        np.save(out_file, np_masks)
    return np_masks

    