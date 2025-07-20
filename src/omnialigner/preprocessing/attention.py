import os
import sys
from typing import Tuple, Dict, List

import dask.array as da
import torch
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import VisionTransformer
from timm.layers import SwiGLUPacked


from omnialigner.dtypes import Dask_image_HWC, Np_image_HWC, Tensor_image_NCHW
from omnialigner.utils.tiles import merge_from_overlapped_tiles
root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))
vendor_path = os.path.join(root_path, 'vendor/istar')
sys.path.append(vendor_path)
from omnialigner.vendor.istar.hipt_4k import transforms
from omnialigner.vendor.istar.extract_features import patchify, rearrange
from omnialigner.vendor.istar.hipt_model_utils import get_vit256

dir_path = os.path.join(os.path.dirname(__file__), "../vit_checkpoints/")



def patchify_dask(x: Dask_image_HWC, patch_size: int) -> Tuple[np.ndarray, Dict]:
    shape_ori = np.array(x.shape[:2])
    shape_ext = ((shape_ori + patch_size - 1) // patch_size) * patch_size
    
    pad_width = (
        (0, shape_ext[0] - x.shape[0]),
        (0, shape_ext[1] - x.shape[1]),
        (0, 0)
    )
    x_padded = da.pad(x, pad_width, mode='edge')
    tiles_shape = shape_ext // patch_size
    x_rechunked = x_padded.rechunk((patch_size, patch_size, -1))
    
    x_patches = x_rechunked.reshape((tiles_shape[0], patch_size, tiles_shape[1], patch_size, -1))
    x_patches = x_patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, x.shape[2]).compute()
    
    shapes = dict(
        original=tuple(shape_ori),
        padded=tuple(shape_ext),
        tiles=tuple(tiles_shape)
    )
    
    return x_patches, shapes

    
class TileDataset(Dataset):
    def __init__(self, tiles: da.Array, transform_eval: transforms=None):
        self.tiles = tiles
        if transform_eval is None:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            # For uint8 input [0,255]:
            # 1. ToTensor: divides by 255 -> [0,1] float32
            # 2. Normalize with mean=0.5, std=0.5:
            #    (x-0.5)/0.5 = 2x-1 -> [-1,1] float32
            transform_eval = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            )
        self.transforms = transform_eval
        
    def __len__(self):
        return len(self.tiles)
        
    def __getitem__(self, idx):
        return self.transforms(self.tiles[idx])
    

class TimmVit(torch.nn.Module):
    """
    TimmVit class definition
    """
    def __init__(self, model: VisionTransformer):
        super().__init__()
        self.model = model
    
    def get_intermediate_layers(self, x, **kwargs):
        """
        Get intermediate layers from the model.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Combined intermediate features.
        """
        x, intermediates = self.model.forward_intermediates(
            x,
            indices=1,
            return_prefix_tokens=1,
            norm=True,
            output_fmt='NCHW',
            intermediates_only=False,
            **kwargs
        )
        feat_intermediate_0_0 = intermediates[0][0]
        feat_intermediate_0_1 = intermediates[0][1]
        feat_intermediate_0_0_flat = feat_intermediate_0_0.flatten(start_dim=2).permute(0, 2, 1)
        feat_combined = torch.cat((feat_intermediate_0_0_flat, feat_intermediate_0_1), dim=1)

        return feat_combined

default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class AttentionModel(torch.nn.Module):
    """
    AttentionModel class definition
    """
    def __init__(
        self,
        model_name=None,
        patch_size = (256, 256),
        subpatch_size = (16, 16),
        device256=default_device,
    ):
        super().__init__()
        self.device256 = device256
        self.model_name = model_name
        self.patch_size = patch_size
        self.subpatch_size = subpatch_size
        self.model256, self.transforms = self._initialize_model_and_transforms(model_name)
        self.model256 = self.model256.to(device256)

    def _initialize_model_and_transforms(self, model_name):
        """
        Initialize model and transforms based on the model name.

        Args:
            model_name (str): Name of the model.

        Returns:
            tuple: Model and transforms.
        """
        if model_name == "hipt":
            model_path = os.path.join(dir_path, 'vit256_small_dino.pth')
            model = get_vit256(pretrained_weights=model_path)
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            base_transforms =  transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=mean, std=std)
            ])
            
        elif model_name == "gigapath":
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
            dict_params = torch.load(os.path.join(dir_path, 'gigapath_pytorch_model.bin'))
            model.load_state_dict(dict_params)
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            base_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        elif model_name == "virchow":
            model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            model.load_state_dict(torch.load(os.path.join(dir_path, 'virchow_pytorch_model.bin')))
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            base_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            
        elif model_name == "UNI":
            model = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            model.load_state_dict(torch.load(os.path.join(dir_path, 'uni_pytorch_model.bin')))
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            base_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        
        elif model_name == "CONCH":
            from conch.open_clip_custom import create_model_from_pretrained
            checkpoint_conch = os.path.join(dir_path, 'conch_pytorch_model.bin')
            model_, base_transforms = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_conch)
            model = model_.visual.trunk
            
            
        else:
            raise ValueError(f"Model {model_name} not found")
   
        wsi_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    base_transforms,
        ])
        if model_name in ("gigapath", "virchow", "UNI"):
            model = TimmVit(model)
        
        return model, wsi_transforms

    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        """
        Prepare image tensor by making its dimensions divisible by the patch size.

        Args:
            img (torch.Tensor): Input image tensor.
            patch_size (int): Desired patch size.

        Returns:
            tuple: New image tensor and patch dimensions.
        """
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        b, c, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)
        return img_new, w_256, h_256

    def process_in_batches(self, data, batch_size, **kwargs):
        """
        Process data in batches using the given model.

        Args:
            data (torch.Tensor): The input data to process.
            batch_size (int): The size of each batch.
            **kwargs: Additional arguments to pass to the model's method.

        Returns:
            torch.Tensor: The concatenated output from all batches.
        """
        data = data.to(self.device256, non_blocking=True)
        output_list = []
        for i in range(0, data.size(0), batch_size):
            batch = data[i:i + batch_size]
            with torch.no_grad():
                feat_intermediate = self.model256.get_intermediate_layers(batch, **kwargs)
                if self.model_name == "hipt" or self.model_name == "CONCH":
                    feat_intermediate = feat_intermediate[0]
                
                output_list.append(feat_intermediate)
        
        output = torch.cat(output_list, dim=0).unsqueeze(0)
        return output

    def forward_all256(self, x):
        """
        Forward pass for all 256x256 patches.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Class and subpatch features.
        """
        batch_256, w_256, h_256 = self.prepare_img_tensor(
            x, patch_size=self.patch_size[0]
        )
        batch_256 = batch_256.unfold(2, self.patch_size[0], self.patch_size[1]).unfold(
            3, self.patch_size[0], self.patch_size[1]
        )
        batch_256 = rearrange(
            batch_256, "b c p1 p2 w h -> (b p1 p2) c w h"
        )

        features_cls256 = []
        features_sub256 = []
        for mini_bs in range(
            0, batch_256.shape[0], self.patch_size[0]
        ):
            minibatch_256 = batch_256[mini_bs:mini_bs + self.patch_size[0]].to(
                self.device256, non_blocking=True
            )
            kwargs = {"batch_size" : 8}
            feat_intermediate = self.process_in_batches(minibatch_256, **kwargs)
            fea_all256 = feat_intermediate[0].cpu()
            fea_cls256 = fea_all256[:, 0]
            fea_sub256 = fea_all256[:, 1:]
            features_cls256.append(fea_cls256)
            features_sub256.append(fea_sub256)

        features_cls256 = torch.vstack(features_cls256)
        features_sub256 = torch.vstack(features_sub256)
        features_cls256 = (
            features_cls256.reshape(w_256, h_256, -1)
            .transpose(0, 1)
            .transpose(0, 2)
            .unsqueeze(dim=0)
        )
        features_sub256 = (
            features_sub256.reshape(w_256, h_256, self.subpatch_size[0], self.subpatch_size[1], -1)
            .permute(4, 0, 1, 2, 3)
            .unsqueeze(dim=0)
        )
        return features_cls256, features_sub256


def get_embeddings_sub_attention(model: AttentionModel, x: np.ndarray):
    # x = x.astype(np.float32) / 255.0
    x = model.transforms(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def get_embeddings_sub(model: AttentionModel, x: Tensor_image_NCHW) -> Tuple[np.ndarray, np.ndarray]:
    # Handle both DataParallel and regular model cases
    if isinstance(model, torch.nn.DataParallel):
        x = x.to(model.device_ids[0])
        x_cls, x_sub = model(x)  
    else:
        device = next(model.parameters()).device
        x = x.to(device)
        x_cls, x_sub = model(x)
     
        
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def get_embeddings_dino(img: np.ndarray,
                        model_name: str="hipt",
                        device: torch.device=torch.device("cpu"),
                        tile_size: int=4096,
                        patch_size: tuple=(256, 256),
                        subpatch_size: tuple=(16, 16),
                        model: torch.nn.Module=None,
    ):
    """
    Get embeddings for debugging.

    Args:
        img (np.ndarray): Input image.
        patch_size (tuple): Patch size.
        subpatch_size (tuple): Subpatch size.
        pretrained (bool): Whether to use pretrained model.
        device (str): Device to use.
        model (torch.nn.Module): Model to use.
        model_name (str): Name of the model.

    Returns:
        tuple: Mid and subpatch embeddings.
    """
    tile_size = 4096
    tiles, shapes = patchify(img, patch_size=tile_size)

    patch_size = (256, 256)
    subpatch_size = (16, 16)
    if model_name in ("UNI", "virchow"):
        patch_size = (224, 224)
        subpatch_size = (16, 16)
    if model_name in ("gigapath"):
        patch_size = (224, 224)
        subpatch_size = (14, 14)
    if model_name in ("CONCH"):
        patch_size = (448, 448)
        subpatch_size = (16, 16)


    if model is None:
        model = AttentionModel(
                model_name=model_name,
                patch_size=patch_size,
                subpatch_size=subpatch_size,
                device256=device)
        model.eval()

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        if i % 10 == 0:
            print('tile', i, '/', len(tiles))

        x_mid, x_sub = get_embeddings_sub_attention(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)

    emb_mid = rearrange(
            emb_mid, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=shapes['tiles'][0], w1=shapes['tiles'][1])

    shape_orig = np.array(shapes['original']) // subpatch_size
    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
                np.array([e[..., i] for e in emb_sub]),
                '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
                h1=shapes['tiles'][0], w1=shapes['tiles'][1])
        chans_sub.append(chan)

    pt_emb_mid = torch.from_numpy(np.array(emb_mid))
    pt_emb_sub = torch.from_numpy(np.array(chans_sub))
    return pt_emb_mid, pt_emb_sub



def _calculate_embedding(
        model: AttentionModel,
        tile_loader: DataLoader
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    emb_sub = []
    emb_mid = []
    for i, batch_tiles in enumerate(tile_loader):
        if i % 2 == 0:
            print(f'Processing batch {i}/{len(tile_loader)}')
            
        batch_mid, batch_sub = get_embeddings_sub(model, batch_tiles)
        emb_mid.extend(batch_mid)
        emb_sub.extend(batch_sub)
    
    return emb_sub, emb_mid


def _dask_to_tileDataLoader(
        img: Np_image_HWC|Dask_image_HWC,
        tile_size: int=4096,
        batch_size: int=4,
        num_workers: int=0,
        transform_eval: transforms=None
    ) -> Tuple[DataLoader, Dict]:
    if isinstance(img, np.ndarray):
        tiles, shapes = patchify(img, patch_size=tile_size)
    
    tiles, shapes = patchify_dask(img, patch_size=tile_size)
    tile_dataset = TileDataset(tiles, transform_eval=transform_eval)
    tile_loader = DataLoader(tile_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    return tile_loader, shapes

def _process_emb_sub(emb_sub, shapes) -> da.Array:
    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
                np.array([e[..., i] for e in emb_sub]),
                '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
                h1=1, w1=1)
        chans_sub.append(chan)

    return da.from_array(np.moveaxis(np.array(chans_sub), 0, 2))

def _process_emb_mid(emb_mid, shapes) -> da.Array:
    emb_mid = rearrange(
            emb_mid, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=1, w1=1
    )  # 16 x 16 x 384
    return da.from_array(emb_mid)

def _merge_emb(l_tile_coords, tag, tmp_prefix):
    n_tiles = len(l_tile_coords)
    l_tiles = []
    l_emb = []
    for i in range(n_tiles):
        emb = da.from_zarr(f"{tmp_prefix}_{i}.{tag}.zarr")
        tile_coord = l_tile_coords[i]
        h, w, _ = emb.shape
        l_tiles.append([tile_coord[0]*h, tile_coord[1]*w, tile_coord[2]*h, tile_coord[3]*w])
        emb = torch.from_numpy( emb[np.newaxis,:,:,:].compute()) 
        l_emb.append(emb)

    out_arr = merge_from_overlapped_tiles(l_tiles_extend=l_tiles, l_disp=l_emb)
    return out_arr
    

def get_embeddings_dino_v2(img: da.Array,
                        model_name: str="hipt",
                        device: torch.device=torch.device("cpu"),
                        tile_size: int=4096,
                        patch_size: tuple=(256, 256),
                        subpatch_size: tuple=(16, 16),
                        tmp_prefix: str= None,
                        model: torch.nn.Module=None,
    ):
    """
    Get embeddings for debugging.

    Args:
        img (np.ndarray): Input image.
        patch_size (tuple): Patch size.
        subpatch_size (tuple): Subpatch size.
        pretrained (bool): Whether to use pretrained model.
        device (str): Device to use.
        model (torch.nn.Module): Model to use.
        model_name (str): Name of the model.

    Returns:
        tuple: Mid and subpatch embeddings.
    """
    tile_size = 4096
    # tiles, shapes = patchify(img, patch_size=tile_size)
    tiles, shapes = patchify_dask(img, patch_size=tile_size)

    if model is None:
        model = AttentionModel(
                model_name=model_name,
                patch_size=patch_size,
                subpatch_size=subpatch_size,
                device256=device)
        model.eval()

    emb_sub = []
    emb_mid = []
    l_tile_coords = []
    n_tiles = len(tiles)

    for i in range(n_tiles):
        if i % 10 == 0:
            print('tile', i, '/', n_tiles)

        
        i_row = i // shapes["tiles"][0]
        j_col = i % shapes["tiles"][1]
        tile_coord = [i_row, j_col, i_row+1, j_col+1]
        l_tile_coords.append(tile_coord)

        if os.path.isdir(f"{tmp_prefix}_{i}.sub.zarr"):
            continue
        
        x_mid, x_sub = get_embeddings_sub_attention(model, tiles[i])
        emb_mid = [x_mid]
        emb_sub = [x_sub]

        da_emb_cls = _process_emb_mid(emb_mid, shapes=shapes)
        da_emb_sub = _process_emb_sub(emb_sub, shapes=shapes)
        da.to_zarr(da_emb_cls, f"{tmp_prefix}_{i}.cls.zarr", overwrite=True)
        da.to_zarr(da_emb_sub, f"{tmp_prefix}_{i}.sub.zarr", overwrite=True)

    da_emb_cls_all = _merge_emb(l_tile_coords, "cls", tmp_prefix)
    da_emb_sub_all = _merge_emb(l_tile_coords, "sub", tmp_prefix)
    da_emb_cls_all = da.from_array(da_emb_cls_all.numpy())
    da_emb_sub_all = da.from_array(da_emb_sub_all.numpy())
    return da_emb_cls_all[0].rechunk((32, 512, 512)), da.moveaxis(da_emb_sub_all[0], 2, 0).rechunk((32, 512, 512))

def run_vit_attention(tiff, generate_WSI_func, model_name="UNI", device=torch.device("cpu")):
    print(f'Extracting embeddings {model_name}...')

    da_arr = generate_WSI_func(tiff=tiff)
    wsi = da_arr.compute()
    pt_emb_cls, pt_emb_sub = get_embeddings_dino(wsi, model_name=model_name, device=device)

    return {"cls": pt_emb_cls, "sub": pt_emb_sub}



