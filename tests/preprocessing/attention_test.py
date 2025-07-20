import unittest
import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch
import dask.array as da

from omnialigner.preprocessing.attention import patchify_dask, patchify, _dask_to_tileDataLoader, get_embeddings_dino, get_embeddings_dino_v2


root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), './'))
dict_config = {
    "tile_size": 4096,
    "hipt": {
        "patch_size": [256, 256],
        "subpatch_size": [16, 16]
    },
    "UNI": {
        "patch_size": [224, 224],
        "subpatch_size": [16, 16]
    },
    "virchow": {
        "patch_size": [224, 224],
        "subpatch_size": [16, 16]
    },
    "gigapath": {
        "patch_size": [224, 224],
        "subpatch_size": [14, 14]
    },
    "CONCH": {
        "patch_size": [448, 448],
        "subpatch_size": [16, 16]
    },
    
}


class TestAttentionModel(unittest.TestCase):
    def test_attention_model(self):
        img = np.array(Image.open(f"{root_path}/conch_roi1.jpg"))
        img = cv2.resize(img, (5000, 5000))
        pt_cls, pt_emb = get_embeddings_dino(img, model_name="hipt", device=torch.device("cpu"))
        da_cls, da_emb = get_embeddings_dino_v2(img, model_name="hipt", device=torch.device("cpu"), tmp_prefix="./conch_roi1")
        diff_cls = np.mean(da_cls.compute() - pt_cls.numpy())
        diff_sub = np.mean(da_emb.compute() - pt_emb.numpy())
        self.assertLess(diff_cls, 1e-5)
        self.assertLess(diff_sub, 1e-5)
        print("test_attention_model done")

    def test_patchify(self):
        l_examples = [(2048, 2048, 256), (2049, 2048, 256), (2048, 2049, 256)]
        for H_out, W_out, patch_size in tqdm(l_examples, desc="test_patchify"):
            x = np.random.rand(H_out, W_out, 3)
            x_da = da.from_array(x, chunks=(64, 64, 3))
            patches, shapes = patchify_dask(x_da, patch_size)
            patches_expected, shapes_expected = patchify(x, patch_size)
            diff = (patches - np.stack(patches_expected, axis=0)).mean()
            self.assertLess(diff, 1e-5)

            assert sum(shapes['original'] != shapes_expected['original']) == 0
            assert sum(shapes['padded'] != shapes_expected['padded']) == 0
            assert sum(shapes['tiles'] != shapes_expected['tiles']) == 0

        print("test_patchify done")

    def test_image(self):
        H_out = 2048
        W_out = 2048
        pseudo_image = da.from_array(
            np.random.randint(0, 256, size=(H_out, W_out, 3), dtype=np.uint8),
            chunks=(1, 1, 3)
        )
        tile_loader, _ = _dask_to_tileDataLoader(
            pseudo_image,
            tile_size=256,
        )
        l_tiles = [ t for t in tqdm(tile_loader, desc="test_image")]
        raw_dask_np = (((pseudo_image[0:256, 0:256, :] / 255) - 0.5) / 0.5).compute()
        tile_tensor_np = l_tiles[0][0].permute(1,2,0).numpy()
        diff = np.mean( np.abs( raw_dask_np - tile_tensor_np ) )
        self.assertLess(diff, 1e-5)


        raw_dask_np = (((pseudo_image[0:256, 256:512, :] / 255) - 0.5) / 0.5).compute()
        tile_tensor_np = l_tiles[0][1].permute(1,2,0).numpy()
        diff = np.mean( np.abs( raw_dask_np - tile_tensor_np ) )
        self.assertLess(diff, 1e-5)
        print("test_image done")


if __name__ == "__main__":
    unittest.main()
    # test_patchify()
    # test_image()
    # test_attention_model()