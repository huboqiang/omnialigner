from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc

from omnialigner.dtypes import Tensor_image_NCHW, Tensor_l_kpt_pair, AnnData_l_cells_pair, Tensor_cells_N_xy, Tensor_cells_N_embed, Np_l_cells_MNN
from omnialigner.utils.mnn import calculate_overlapped_mnn_pairs


class OverlappedImageLayerDataset(Dataset):
    def __init__(
        self, 
        image_3d_tensor: Tensor_image_NCHW = None, 
        l_kpt_pairs: Tensor_l_kpt_pair = None, 
        anndatas_cells: AnnData_l_cells_pair = None, 
        batch_size: int = 32,
        max_size: int = 1280,
        overlap: int = 8,
        random_cells: int = 3_000
    ):
        """
        Overlapped Image Layer Dataset for batching 3D image tensors with overlapping layers.

        image_3d_tensor, l_kpt_pairs, anndatas_cells cannot all be None at the same time.

        Args:
            image_3d_tensor (Tensor_image_NCHW, optional): 3D image tensor of shape (N, C, H, W).
            l_kpt_pairs (Tensor_l_kpt_pair, optional): List of keypoint pairs between consecutive layers. 
                Defaults to None.
            anndatas_cells (AnnData_l_cells_pair, optional): List of AnnData objects for each layer. 
                Defaults to None.
            batch_size (int, optional): Number of layers per batch. Defaults to 32.
            overlap (int, optional): Number of overlapping layers between batches. Defaults to 8.
            random_cells (int, optional): Number of cells to randomly sample per layer. 
                If -1, use all cells. Defaults to -1.
        
        __getitem__ returns:
            image_batch (Tensor_image_NCHW): Batch of image layers.
            kpt_pairs_batch (Tensor_l_kpt_pair): Keypoint pairs for the batch.
            layer_indices (Tensor): Indices of the layers in the batch.
            l_cell_pos (List[Tensor_cells_N_xy]): List of cell positions for each layer in the batch.
            l_cell_emb (List[Tensor_cells_N_embed]): List of cell embeddings for each layer in the batch.
            mnn_cells (List[List[Tuple[int, int]]]): MNN cell pairs for the batch.
        """
        self.batch_size = batch_size
        self.overlap = overlap
        self.random_cells = random_cells
        self.max_size = max_size
        if image_3d_tensor is not None:
            self.total_images = image_3d_tensor.shape[0]
        elif l_kpt_pairs is not None:
            self.total_images = len(l_kpt_pairs) + 1
        elif anndatas_cells is not None:
            self.total_images = len(anndatas_cells)
        else:
            raise ValueError("At least one of image_3d_tensor, l_kpt_pairs, or anndatas_cells must be provided.")
        
        
        if image_3d_tensor is None:
            image_3d_tensor = torch.zeros((self.total_images, 1, self.max_size, self.max_size), dtype=torch.float32)

        if l_kpt_pairs is None:
            l_kpt_pairs = [ [] for _ in range(self.total_images-1)]

        self.image_3d_tensor = image_3d_tensor
        self.l_kpt_pairs = l_kpt_pairs
        self.anndatas_cells = anndatas_cells if anndatas_cells is not None else [ None for _ in range(self.total_images) ]
        mnn_cells = [ [] for _ in range(self.total_images-1) ]
        
        self.mnn_cells = mnn_cells

    def __len__(self):
        return int( (self.total_images - self.overlap) // (self.batch_size - self.overlap + 0.1) + 1 )

    def __getitem__(self, idx) -> Tuple[Tensor_image_NCHW, Tensor_l_kpt_pair, torch.Tensor, List[Tensor_cells_N_xy], List[Tensor_cells_N_embed], Np_l_cells_MNN]:
        start_idx = idx * (self.batch_size - self.overlap)
        end_idx = start_idx + self.batch_size
        end_idx = min(end_idx, self.total_images)
        
        image_out = None if self.image_3d_tensor is None else self.image_3d_tensor[start_idx:end_idx]
        l_kpt_out = None if self.l_kpt_pairs is None else self.l_kpt_pairs[start_idx:end_idx-1]
        
        # Convert AnnData list to tensor lists (similar to _prepare_device in aligner.py)
        
        l_cell_pos = []
        l_cell_emb = []
        for i_layer in range(start_idx, end_idx):
            if i_layer >= len(self.anndatas_cells):
                print("Layer index exceeds available AnnData layers:", i_layer)
                print(self.anndatas_cells)

            anndata_cells = self.anndatas_cells[i_layer]
            if anndata_cells is None:
                l_cell_pos.append([])
                l_cell_emb.append([])
                continue

            # Random sampling if needed
            if self.random_cells > 0 and anndata_cells.n_obs > self.random_cells:
                n_i = min(self.random_cells, anndata_cells.n_obs)
                idx_i = np.random.choice(anndata_cells.n_obs, n_i, replace=False)
                anndata_cells = anndata_cells[idx_i, :]
            
            # Extract embeddings
            embed = anndata_cells.X
            if hasattr(anndata_cells.X, 'toarray'):
                embed = anndata_cells.X.toarray()
            
            # Extract spatial coordinates
            spatial = anndata_cells.obsm["spatial"]
            
            # Convert to tensors (CPU, will be moved to device in model)
            cells_emb = torch.FloatTensor(embed).float()
            cells_pos = torch.FloatTensor(spatial).float() / self.max_size
            l_cell_pos.append(cells_pos)                
            l_cell_emb.append(cells_emb)

        
        mnn_cells = None if self.mnn_cells is None else self.mnn_cells[start_idx:end_idx-1]
        
        return image_out, l_kpt_out, torch.arange(start_idx, end_idx), l_cell_pos, l_cell_emb, mnn_cells

