import unittest
import cv2
import torch
import numpy as np
from ...preprocessing.attention import get_embeddings_dino
from ...datasets.datasets import generate_WSI
from ...vendor.istar.extract_features import get_embeddings



def test_vit_attention(model_name="UNI"):
    print(f'Extracting embeddings {model_name}...')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    generate_WSI_func = generate_WSI("ANHIR")
    tiff = "/cluster/home/bqhu_jh/projects/SG_0623/tests/models/COAD_01-HE.ome.tiff"
    da_arr = generate_WSI_func(tiff=tiff)
    wsi = da_arr.compute()
    patch_size = (256, 256)
    subpatch_size = (16, 16)
    if model_name in ("UNI", "virchow"):
        patch_size = (224, 224)
        subpatch_size = (16, 16)
    if model_name in ("gigapath"):
        patch_size = (224, 224)
        subpatch_size = (14, 14)

    pt_emb_cls, pt_emb_sub = get_embeddings_dino(
                        wsi, patch_size=patch_size, subpatch_size=subpatch_size, pretrained=True, model_name=model_name,
                        device=device)

    np_sub = torch.mean(pt_emb_sub, axis=0).numpy()
    cv2.imwrite(f"./tmp_{model_name}.png", (255*(np_sub-np_sub.min()) / (np_sub.max()-np_sub.min())).astype(np.uint8))
    return pt_emb_cls, pt_emb_sub


class TestTimmVitAttentions(unittest.TestCase):
    def test_vit_attention_hipt(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        generate_WSI_func = generate_WSI("ANHIR")
        tiff = "/cluster/home/bqhu_jh/projects/SG_0623/tests/models/COAD_01-HE.ome.tiff"
        da_arr = generate_WSI_func(tiff=tiff)
        wsi = da_arr.compute()
        pt_emb_cls, pt_emb_sub = get_embeddings_dino(
                            wsi, pretrained=True,
                            device=device)

        pt_emb_cls_target, pt_emb_sub_target = get_embeddings(
                            wsi, pretrained=True,
                            device=device)
        
        pt_emb_sub = torch.from_numpy(np.array(pt_emb_sub))
        pt_emb_sub_target = torch.from_numpy(np.array(pt_emb_sub_target))
        _, H, W = pt_emb_sub_target.shape
        self.assertAlmostEqual(torch.mean(pt_emb_sub_target - pt_emb_sub[:, 0:H, 0:W]).item(), 0.)


    # def test_vit_attention_timm(self):
    #     for model_name in ["UNI", "gigapath", "virchow"]:
    #         pt_emb_cls, pt_emb_sub = test_vit_attention(model_name=model_name)
    #         expected_shapes = {
    #             "UNI": torch.Size([784, 1152, 576]),
    #             "gigapath": torch.Size([1536, 1008, 504]),
    #             "virchow": torch.Size([1280, 1152, 576])
    #         }
    #         expected_shape = expected_shapes.get(model_name)
    #         if expected_shape is None:
    #             raise ValueError(f"Unexpected model name: {model_name}")

    #         self.assertTrue(pt_emb_sub.shape == expected_shape, 
    #             f"Shape mismatch for {model_name}: expected {expected_shape}, got {pt_emb_sub.shape}"
    #         )

    #         expected_shapes = {
    #             "UNI": torch.Size([72, 36, 1024]),
    #             "gigapath": torch.Size([72, 36, 1536]),
    #             "virchow": torch.Size([72, 36, 1280])
    #         }
    #         expected_shape = expected_shapes.get(model_name)
    #         if expected_shape is None:
    #             raise ValueError(f"Unexpected model name: {model_name}")

    #         self.assertTrue(pt_emb_cls.shape == expected_shape, 
    #             f"Shape mismatch for {model_name}: expected {expected_shape}, got {pt_emb_cls.shape}"
    #         )

if __name__ == "__main__":
    unittest.main()
    # test_vit_attention(model_name="hipt")
    # test_vit_attention(model_name="UNI")
    # test_vit_attention(model_name="gigapath")
    # test_vit_attention(model_name="virchow")