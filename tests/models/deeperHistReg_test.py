import os
import unittest
import sys

import cv2
import numpy as np
import torch
import pytorch_lightning as pl

from omnialigner.configs.config import load_config
from omnialigner.align.models.grid_2d.deeperhistreg_module import image_aligner_DeeperHistReg
from omnialigner.align.models.align_2d import SG2DAligner
import omnialigner.plotting as viz

curr_path = os.path.dirname(os.path.abspath(__file__))

def run_nonrigid_deeperhistreg(image_F, image_moved, nonrigid_registration_params):
    res, disp = image_aligner_DeeperHistReg(
                        image_F,
                        image_moved,
                        nonrigid_registration_params=nonrigid_registration_params
    )
    return res



def run_nonrigid_omnialigner(image_F, image_moved, l_kpt_pairs):
    
    H, W = image_F.shape[0], image_F.shape[1]
    tensor_F = viz.image_viz.im2tensor(image_F)
    tensor_moved = viz.image_viz.im2tensor(image_moved)
    image_3d_tensor = torch.cat([tensor_F[:, 0:1, :, :], tensor_moved[:, 0:1, :, :]], dim=0)

    dict_config = load_config(f"{curr_path}/config_2D_deeperhistreg.yaml")
    model_deep = SG2DAligner(
            image_3d_tensor=image_3d_tensor,
            l_kpt_pairs=l_kpt_pairs,
            l_init_trs= torch.FloatTensor([0,0,0,0,0]),
            dict_config=dict_config,
            model_type="nonrigid",
            save_prefix="DeeperHistReg"
    )

    trainer = pl.Trainer(max_epochs=9, gpus=1)
    trainer.fit(model_deep)

    model_deep.grid2d_model.set_device(torch.device("cpu"))
    model_deep.grid2d_model.tensor_size = [H, W]
    image_M_to_F, output_kpt = model_deep.grid2d_model.forward(tensor_moved, None)
    return image_M_to_F


def calculate_loss(image_M_to_F, res):
    loss_l1 = torch.nn.functional.l1_loss(image_M_to_F, res)
    sim = torch.nn.functional.cosine_similarity(image_M_to_F, res, dim=1)
    loss_cossim = torch.mean( sim[sim>0] )
    return loss_l1, loss_cossim


def run_test_nonrigid(input_pt):
    image_F = input_pt["image_F"]
    image_moved = input_pt["image_moved"]
    l_kpt_pairs = input_pt["l_kpt_pairs"]
    nonrigid_registration_params = input_pt["nonrigid_registration_params"]
    res = run_nonrigid_deeperhistreg(image_F, image_moved, nonrigid_registration_params)
    image_M_to_F = run_nonrigid_omnialigner(image_F, image_moved, l_kpt_pairs)
    loss_l1, loss_cossim = calculate_loss(image_M_to_F, res)
    return res, image_M_to_F


class TestDeeperHistRegNonrigid(unittest.TestCase):
    def test_affine(self):
        input_pt = torch.load(f"{curr_path}/inputs.pt")
        image_F = input_pt["image_F"]
        image_M = input_pt["image_M"]
        image_moved = input_pt["image_moved"]
        l_kpt_pairs = input_pt["l_kpt_pairs"]
        nonrigid_registration_params = input_pt["nonrigid_registration_params"]


        H, W = image_F.shape[0], image_F.shape[1]
        tensor_F = viz.image_viz.im2tensor(image_F)
        tensor_M = viz.image_viz.im2tensor(image_M)
        tensor_moved = viz.image_viz.im2tensor(image_moved)
        image_3d_tensor = torch.cat([tensor_F, tensor_M], dim=0)
        config_dict = load_config(f"{curr_path}/config_2D_deeperhistreg.yaml")
        model = SG2DAligner(
                image_3d_tensor=image_3d_tensor,
                l_kpt_pairs=l_kpt_pairs,
                l_init_trs= torch.zeros(5),
                dict_config=config_dict,
                model_type="affine",
                save_prefix="affine"
        )

        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(model)

        model.grid2d_model.tensor_size = [H, W]
        image_M_to_F, output_kpt = model.grid2d_model.forward(viz.image_viz.im2tensor(image_M), None)

        image_moved_ = viz.image_viz.tensor2im(image_M_to_F)
        np_out = image_F.copy()
        np_out[:, :, 1] = image_moved_[:,:,0]
        np_out[:, :, 2] = 0
        cv2.imwrite("result_affine.png", np_out[:, :, ::-1])
        loss_l1 = torch.nn.functional.l1_loss(image_M_to_F, tensor_moved)
        self.assertLessEqual(loss_l1, 0.001)

        
    def test_nonrigid(self):
        input_pt = torch.load(f"{curr_path}/inputs.pt")
        image_F = input_pt["image_F"]
        image_M = input_pt["image_M"]
        image_moved = input_pt["image_moved"]
        l_kpt_pairs = input_pt["l_kpt_pairs"]

        res, image_M_to_F  = run_test_nonrigid(input_pt=input_pt)
        loss_l1, loss_cossim = calculate_loss(image_M_to_F, res)

        np_out1 = image_F.copy()
        np_out1[:, :, 1] = image_M[:,:,0]
        np_out1[:, :, 2] = 0

        np_out2 = image_F.copy()
        np_out2[:, :, 1] = image_moved[:,:,0]
        np_out2[:, :, 2] = 0

        np_out3 = image_F.copy()
        np_out3[:, :, 1] = viz.image_viz.tensor2im( res.cpu() )[:,:,0]
        np_out3[:, :, 2] = 0

        np_out4 = image_F.copy()
        np_out4[:, :, 1] = viz.image_viz.tensor2im( image_M_to_F.cpu() )[:,:,0]
        np_out4[:, :, 2] = 0

        cv2.imwrite("result.png", np.hstack([np_out1, np_out2, np_out3, np_out4])[:, :, ::-1])

        self.assertLessEqual(loss_l1, 0.05)
        self.assertLessEqual(1-loss_cossim, 0.02)



if __name__ == '__main__':
    unittest.main()
    # debug_nonrigid()

