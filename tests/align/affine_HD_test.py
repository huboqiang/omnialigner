import os
import unittest

import cv2
import torch
import yaml
import numpy as np
import dask.array as da
from dask_image.ndinterp import affine_transform

import omnialigner as om
from omnialigner.preprocessing.pad import get_pad_size
from omnialigner.align import apply_image_HD
from omnialigner.cache_files import StageSampleTag, StageTag, DataType
from omnialigner.align.nonrigid_HD import Grid2DModelDual, init_nonrigid_model

curr_dir = os.path.dirname(os.path.abspath(__file__))

def dask_affine_transform_multichannel(arr, scale_factor=40, **kwargs):
    """
    Apply affine transform to a dask array with multiple channels.

    Parameters:
    arr (dask.array): Input dask array with shape (H, W, C).
    scale_factor (float): Scale factor for the affine transform.
    **kwargs: Additional keyword arguments for the affine_transform function.

    Returns:
    dask.array: Transformed dask array with shape (H, W, C).
    """
    matrix = np.eye(2) * scale_factor
    offset = (0, 0)
    output_shape = (int(arr.shape[0] / scale_factor), int(arr.shape[1] / scale_factor))

    kwargs["order"] = kwargs.get("order", 1)
    kwargs["mode"] = kwargs.get("mode", 'constant')
    kwargs["cval"] = kwargs.get("cval", 0.0)
    kwargs["preserve_range"] = kwargs.get("preserve_range", True)

    
    results = [
        affine_transform(arr[..., i], matrix=matrix, offset=offset, output_shape=output_shape, **kwargs)
        for i in range(arr.shape[-1])
    ]
    return da.stack(results, axis=-1)

class TestAlignAffineHDCustome(unittest.TestCase):
    def load_omdata(self, group_id="ALL-TLS_100"):
        om_config = f"{curr_dir}/../../config/panlab2d/config_pdac.yaml"  # Path to your YAML configuration file
        with open(om_config, 'r') as f:
            template_string = f.read()
            config_info = yaml.load(template_string, Loader=yaml.FullLoader)
            config_info["datasets"]["group"] = f"{group_id}"
            config_info["datasets"]["root_dir"] = "~/projects/scGaussian3dGen"
            config_info["datasets"]["file_data"] = f"{curr_dir}/../../config/panlab2d/data_ALL-TLS.csv"
            config_info["datasets"]["file_IHC_name"] = f"{curr_dir}/../../config/panlab2d/IHC_layer_info.json"
        
        om_data = om.Omni3D(config_info=config_info)
        return om_data

    def load_params(self, om_data: om.Omni3D, i_layer: int=0, zoom_level: int=3):
        dir_crop = StageSampleTag.RAW.get_file_name(projInfo=om_data.proj_info, i_layer=i_layer, check_exist=False)["dir"] + "/crop"
        sample_name = om_data.proj_info.get_sample_name(i_layer)
        np_coords = [0, 0, 1, 1]
        if os.path.exists(f"{dir_crop}/{sample_name}_coords.npy"):
            np_coords = np.load(f"{dir_crop}/{sample_name}_coords.npy")

        pad_size = get_pad_size(om_data, i_layer, zoom_level=zoom_level)
        ratio = torch.load(StageTag.PAD.get_file_name(om_data.proj_info)["l_ratio"])[i_layer]
        affine_matrix = np.array(
            [
                [ratio, 0, 0],
                [0, ratio, 0],
                [0, 0, 1],
            ]
        )
        
        tensor_tfrs_1 = om_data._load_TFRS_params(i_layer)
        dict_affine_model = torch.load(StageTag.AFFINE.get_file_name(om_data.proj_info)["affine_model"])
        tag = f"grid2d_modules.{i_layer}.tensor_trs"
        if tag not in dict_affine_model:
            tag = f"{i_layer}.tensor_trs"

        tensor_tfrs_2 = dict_affine_model[tag]
        kwargs_pad = {"mode": "constant", "pad_size": pad_size}
        if om_data.proj_info.get_dtype(i_layer) == "HE" and om_data.tag == DataType.RAW:
            kwargs_pad["constant_values"] = 255
        
        
        scale_level = om_data.l_scales[zoom_level] / om_data.l_scales[-1]
        target_pad_size = (om_data.max_size*scale_level, om_data.max_size*scale_level)
        return np_coords, kwargs_pad, affine_matrix, tensor_tfrs_1, tensor_tfrs_2, target_pad_size


    def load_test_params(self, group_id="ALL-TLS_100", i_layer=0, zoom_level=3):
        om_data = self.load_omdata(group_id=group_id)
        return self.load_params(om_data, i_layer=i_layer, zoom_level=zoom_level)
        

    def test_apply_image_HD_raw(self):
        group_id = "ALL-TLS_100"
        l_layers = ["HE", "P1", "P2", "P3", "P4"]
        for i_layer, layer_name in enumerate(l_layers):
            file_tiff = f"~/projects/scGaussian3dGen/analysis/panlab/v1/01.ome_tiff/{group_id}/{layer_name}.ome.tiff"
            da_img = om.tl.read_ome_tiff(file_tiff, i_page=-1, i_level=0, l_channels=[0,1,2])
            out = apply_image_HD(da_img)
            is_none = out is None
            self.assertFalse(is_none)

    def test_apply_image_HD_params(self):
        group_id = "ALL-TLS_100"
        zoom_level = 3
        l_layers = ["HE", "P1", "P2", "P3", "P4"]
        om_data = self.load_omdata(group_id=group_id)
        
        for i_layer, layer_name in enumerate(l_layers):
            file_tiff = f"~/projects/scGaussian3dGen/analysis/panlab/v1/01.ome_tiff/{group_id}/{layer_name}.ome.tiff"
            l_channels = [0, 1, 2] if layer_name == "HE" else range(8)


            _, kwargs_pad, affine_matrix, tensor_tfrs_1, tensor_tfrs_2, target_pad_size = self.load_params(om_data, i_layer=i_layer, zoom_level=zoom_level)

            pad_size = kwargs_pad["pad_size"]
            del kwargs_pad["pad_size"]
            da_img = om.tl.read_ome_tiff(file_tiff, i_page=zoom_level, i_level=0, l_channels=l_channels)
            out = apply_image_HD(da_img, pad_size=pad_size, affine_matrix=affine_matrix,tensor_tfrs_1=tensor_tfrs_1, tensor_tfrs_2=tensor_tfrs_2, target_pad_size=target_pad_size, **kwargs_pad)
            
            da_tar = om.align.apply_affine_HD(om_data, i_layer=i_layer, overwrite_cache=False)
            np_out = out.compute()
            np_tar = da_tar.compute()
            delta = (np_out - np_tar).mean()
            self.assertAlmostEqual(delta, 0, places=4)
            # cv2.imwrite(f"{curr_dir}/{group_id}_{layer_name}_{zoom_level}.png", out.compute()[:, :, 0:3])

    def test_apply_image_HD_crop(self):
        group_id = "ALL-TLS_100"
        zoom_level = 0
        l_layers = ["HE"]
        om_data = self.load_omdata(group_id=group_id)
        
        for i_layer, layer_name in enumerate(l_layers):
            file_tiff_qptiff = f"~/projects/scGaussian3dGen/data/panlab/{group_id}/{layer_name}.qptiff"
            file_tiff_ome = f"~/projects/scGaussian3dGen/analysis/panlab/v1/01.ome_tiff/{group_id}/{layer_name}.ome.tiff"
            
            
            l_channels = [0, 1, 2] if layer_name == "HE" else range(8)
            np_coords, kwargs_pad, affine_matrix, tensor_tfrs_1, tensor_tfrs_2, target_pad_size = self.load_params(om_data, i_layer=i_layer, zoom_level=zoom_level)
            scale_factor = om_data.l_scales[0] / om_data.l_scales[zoom_level]
            
            da_img_data_ = om.tl.read_ome_tiff(file_tiff_qptiff, i_page=0, i_level=0, l_channels=l_channels)
            da_img_data = dask_affine_transform_multichannel(da_img_data_, scale_factor=scale_factor)
            da_img_ome = om.tl.read_ome_tiff(file_tiff_ome, i_page=zoom_level, i_level=0, l_channels=l_channels)
            h, w, c = da_img_data.shape
            h_beg, h_end = int(h*np_coords[0]), int(h*np_coords[2])
            w_beg, w_end = int(w*np_coords[1]), int(w*np_coords[3])
            da_img_data = da_img_data[h_beg:h_end, w_beg:w_end, :]
            self.assertEqual(da_img_ome.shape, da_img_data.shape)

    def test_nonrigid(self):
        group_id = "ALL-TLS_100"
        zoom_level = 3
        l_layers = ["HE", "P1", "P2", "P3", "P4"]
        om_data = self.load_omdata(group_id=group_id)
        
        for i_layer, layer_name in enumerate(l_layers):
            file_tiff = f"~/projects/scGaussian3dGen/analysis/panlab/v1/01.ome_tiff/{group_id}/{layer_name}.ome.tiff"
            l_channels = [0, 1, 2] if layer_name == "HE" else range(8)


            _, kwargs_pad, affine_matrix, tensor_tfrs_1, tensor_tfrs_2, target_pad_size = self.load_params(om_data, i_layer=i_layer, zoom_level=zoom_level)

            grid_model: Grid2DModelDual = init_nonrigid_model(om_data, i_layer)
            disp = grid_model.disp.detach()

            pad_size = kwargs_pad["pad_size"]
            del kwargs_pad["pad_size"]
            da_img = om.tl.read_ome_tiff(file_tiff, i_page=zoom_level, i_level=0, l_channels=l_channels)
            out = apply_image_HD(
                da_img,
                pad_size=pad_size,
                affine_matrix=affine_matrix,tensor_tfrs_1=tensor_tfrs_1,
                tensor_tfrs_2=tensor_tfrs_2,
                target_pad_size=target_pad_size,
                l_disps=[disp],
                **kwargs_pad
            )
            
            da_tar = om.align.apply_nonrigid_HD(om_data, i_layer=i_layer, overwrite_cache=False)
            np_out = out.compute()
            np_tar = da_tar.compute()
            delta = (np_out - np_tar).mean()
            self.assertAlmostEqual(delta, 0, places=4)
            # cv2.imwrite(f"{curr_dir}/{group_id}_{layer_name}_{zoom_level}.png", out.compute()[:, :, 0:3])


if __name__ == "__main__":
    unittest.main()
    # test = TestAlignAffineHDCustome()
    # test.test_apply_image_HD_raw()
    # test.test_apply_image_HD_params()
    # test.test_apply_image_HD_crop()
    # test.test_nonrigid()