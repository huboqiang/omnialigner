import torch
from torch.utils.data import Dataset


class OverlappedImageLayerDataset(Dataset):
    def __init__(self, image_3d_tensor, l_kpt_pairs=None, batch_size=32, overlap=8):
        self.image_3d_tensor = image_3d_tensor
        self.batch_size = batch_size
        self.overlap = overlap
        self.total_images = image_3d_tensor.shape[0]
        if l_kpt_pairs is None:
            l_kpt_pairs = [ [] for _ in range(self.total_images-1)]

        self.l_kpt_pairs = l_kpt_pairs

    def __len__(self):
        return int( (self.total_images - self.overlap) // (self.batch_size - self.overlap + 0.1) + 1 )

    def __getitem__(self, idx):
        start_idx = idx * (self.batch_size - self.overlap)
        end_idx = start_idx + self.batch_size
        end_idx = min(end_idx, self.total_images)
        return self.image_3d_tensor[start_idx:end_idx], self.l_kpt_pairs[start_idx:end_idx-1], torch.arange(start_idx, end_idx)
    
